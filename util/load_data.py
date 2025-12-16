import pickle
import numpy as np
import pandas as pd
from pathlib import Path

def get_time_series_dict(data_location, pkl_path):

    with open(pkl_path, "rb") as f:
        abmelt_data = pickle.load(f)

    # optional removal you had before
    if "crenezumab" in abmelt_data:
        del abmelt_data["crenezumab"]

    TARGET_TEMPS = ["300K", "350K", "400K"]
    TARGET_LEN = 500

    def resample_series(series: np.ndarray, target_len: int) -> np.ndarray:
        orig_len = len(series)
        if orig_len == target_len:
            return series.copy()
        x_old = np.linspace(0, 1, orig_len)
        x_new = np.linspace(0, 1, target_len)
        return np.interp(x_new, x_old, series)

    def infer_temp_from_colname(colname: str):
        for t in TARGET_TEMPS:
            if t in str(colname):
                return t
        return None

    downsampled_dict = {}
    out_dir = Path(data_location) / "abmelt_downsampled_multiindex"
    out_dir.mkdir(parents=True, exist_ok=True)

    for antibody, feat_dict in abmelt_data.items():
        # We'll collect tuples and arrays, then make a MultiIndex from the tuples
        tuples = []      # list of (feature, subfeature, temp)
        arrays = []      # list of 1D numpy arrays length TARGET_LEN
        seen_tuples = set()

        for feature_name, df in feat_dict.items():
            if not isinstance(df, pd.DataFrame):
                continue

            n_rows, n_cols = df.shape
            colnames = list(df.columns)

            # infer temp by column index for 3-column cases (common)
            col_idx_to_temp = {}
            for i, col in enumerate(colnames):
                t = infer_temp_from_colname(col)
                if t is not None:
                    col_idx_to_temp[i] = t
            if len(col_idx_to_temp) == 0 and n_cols == len(TARGET_TEMPS):
                for i, t in enumerate(TARGET_TEMPS):
                    col_idx_to_temp[i] = t

            for i, col in enumerate(colnames):
                col_temp = col_idx_to_temp.get(i, None)
                if col_temp is None:
                    col_temp = infer_temp_from_colname(col)

                if col_temp is not None and col_temp not in TARGET_TEMPS:
                    # skip temps we don't want
                    continue

                # numeric series
                series = df.iloc[:, i].to_numpy(dtype=float)
                orig_len = len(series)

                if orig_len >= 2:
                    arr = resample_series(series, TARGET_LEN)
                else:
                    arr = np.full(TARGET_LEN, np.nan, dtype=float)

                # derive a safe subfeature name (strip temp if present to avoid duplication)
                raw_sub = str(col)
                # remove temperature substrings from end if present, e.g. "..._300K"
                for t in TARGET_TEMPS:
                    if raw_sub.endswith(t):
                        # also strip trailing delimiters like '_' or ' ' or '-'
                        # remove the last occurrence of the temp substring
                        raw_sub = raw_sub[: -len(t)].rstrip("_ -")
                        break
                safe_subfeature = raw_sub.replace(" ", "_")

                temp_label = col_temp if col_temp is not None else "NAtemp"

                # Build candidate tuple
                tup = (feature_name, safe_subfeature, temp_label)

                # Ensure uniqueness: if the same tuple already exists, append a numeric suffix to subfeature
                if tup in seen_tuples:
                    # mutate safe_subfeature to make it unique
                    suffix = 1
                    new_sub = f"{safe_subfeature}__dup{suffix}"
                    new_tup = (feature_name, new_sub, temp_label)
                    while new_tup in seen_tuples:
                        suffix += 1
                        new_sub = f"{safe_subfeature}__dup{suffix}"
                        new_tup = (feature_name, new_sub, temp_label)
                    tup = new_tup
                    safe_subfeature = new_sub

                seen_tuples.add(tup)
                tuples.append(tup)
                arrays.append(arr)

        # create DataFrame with MultiIndex columns
        if len(tuples) == 0:
            df_out = pd.DataFrame(index=np.arange(TARGET_LEN))
        else:
            # MultiIndex from tuples
            mi = pd.MultiIndex.from_tuples(tuples, names=["feature", "subfeature", "temp"])
            # arrays is list of 1D arrays; stack into 2D (columns)
            data2d = np.column_stack(arrays)  # shape (TARGET_LEN, ncols)
            df_out = pd.DataFrame(data2d, index=np.arange(TARGET_LEN), columns=mi)

        downsampled_dict[antibody] = df_out

        # optional: save to CSV (note: MultiIndex columns will be written with multiple header rows)
        csv_path = out_dir / f"{antibody}.csv"
        df_out.to_csv(csv_path, index_label="timepoint")
    
    return downsampled_dict