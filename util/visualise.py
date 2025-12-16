import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde


def _get_series_for_temps(df, feature, subfeature, temps):
    """
    Given a DataFrame (which may have MultiIndex columns or flat columns),
    return an ordered dict mapping temp -> 1D numpy array (values, NaNs allowed).
    """
    from collections import OrderedDict
    out = OrderedDict()

    # detect MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        # columns are (feature, subfeature, temp)
        # try exact match
        for t in temps:
            try:
                series = df[(feature, subfeature, t)]
                out[t] = series.to_numpy(dtype=float)
            except KeyError:
                # try alternative: sometimes subfeature names include temp suffix; try to find matching tuples
                matches = [col for col in df.columns if col[0]==feature and col[1].startswith(subfeature) and col[2]==t]
                if matches:
                    out[t] = df[matches[0]].to_numpy(dtype=float)
                else:
                    out[t] = None
    else:
        # flat columns like "angles__HL_angle_all__300K" or "angles__HL_angle_all_300K"
        for t in temps:
            # three name patterns to try
            flat1 = f"{feature}__{subfeature}__{t}"
            flat2 = f"{feature}__{subfeature}_{t}"
            flat3 = f"{feature}__{subfeature}{t}"
            # also try with hyphen
            flat4 = f"{feature}__{subfeature}__{t.replace('K','K')}"
            found = None
            for name in (flat1, flat2, flat3, flat4):
                if name in df.columns:
                    found = name
                    break
            if found is not None:
                out[t] = df[found].to_numpy(dtype=float)
            else:
                # try fuzzy match: any column containing feature and subfeature and temp
                cand = [c for c in df.columns if feature in str(c) and subfeature in str(c) and t in str(c)]
                if cand:
                    out[t] = df[cand[0]].to_numpy(dtype=float)
                else:
                    out[t] = None
    return out


def plot_subfeature(downsampled_dict,
                    antibody,
                    feature,
                    subfeature,
                    temps=("300K", "350K", "400K"),
                    figsize=(12,5),
                    linewidth=1.2,
                    kde_bandwidth=None,
                    hist_bins=60,
                    cmap=None,
                    show=True):
    """
    Plot time series and value KDEs for a given feature/subfeature at specified temps.

    Parameters
    ----------
    downsampled_dict : dict
        dict of DataFrames produced earlier (key = antibody).
    antibody : str
        antibody key to select DataFrame from downsampled_dict.
    feature : str
        feature level (e.g., 'angles').
    subfeature : str
        subfeature name (e.g., 'HL_angle_all' or 'HL_angle_all' depending on naming).
    temps : sequence of str
        list/tuple of temperatures in desired order (defaults to ('300K','350K','400K')).
    figsize : tuple
        matplotlib figure size.
    linewidth : float
        line width for time series.
    kde_bandwidth : float or None
        If given, passed to gaussian_kde as bw_method (float multiplier). None uses default.
    hist_bins : int
        if KDE fails, fallback to histogram with this many bins.
    cmap : matplotlib colormap or None
        colormap for temperature lines. If None, uses default color cycle.
    show : bool
        whether to call plt.show() before returning the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if antibody not in downsampled_dict:
        raise KeyError(f"Antibody '{antibody}' not found in downsampled_dict")

    df = downsampled_dict[antibody]

    series_map = _get_series_for_temps(df, feature, subfeature, temps)

    # prepare figure with two axes (time series left, KDE right)
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios":[2,1]})
    ax_ts, ax_kde = axes

    # color setup
    if cmap is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    else:
        cmap = plt.get_cmap(cmap)
        colors = [cmap(i / max(1, len(temps)-1)) for i in range(len(temps))]

    plotted_any = False
    for i, t in enumerate(temps):
        arr = series_map.get(t, None)
        color = colors[i % len(colors)]
        label = f"{t}"

        if arr is None:
            ax_ts.plot([], [], color=color, label=f"{label} (missing)")
            continue

        # drop NaNs for plotting/KDE
        valid_mask = ~np.isnan(arr)
        if valid_mask.sum() == 0:
            ax_ts.plot([], [], color=color, label=f"{label} (all NaN)")
            continue

        ts_vals = arr[valid_mask]
        # time axis (0..len-1). We'll align to the full TARGET_LEN if DataFrame has index
        if hasattr(df, "index") and len(df.index) == len(arr):
            x = df.index.to_numpy()
        else:
            x = np.arange(len(arr))

        # time series (plot full arr but mask NaNs)
        ax_ts.plot(x, arr, label=label, linewidth=linewidth, color=color, alpha=0.9)
        plotted_any = True

        # KDE / histogram on right axis: use ts_vals
        # first check if we have enough unique points for KDE
        try:
            if  np.unique(ts_vals).size > 1:
                kde = gaussian_kde(ts_vals, bw_method=kde_bandwidth)
                # evaluate kde on a sensible grid
                vmin, vmax = np.percentile(ts_vals, [0.5, 99.5])
                if vmin == vmax:
                    vmin, vmax = ts_vals.min(), ts_vals.max()
                grid = np.linspace(vmin, vmax, 512)
                dens = kde(grid)
                # normalize densities so different temps are comparable visually
                dens = dens / dens.max() if dens.max() > 0 else dens
                # shift each density horizontally slightly by index to avoid total overlap (optional)
                ax_kde.plot(dens, grid, color=color, linewidth=1.6, label=label)
                # median line
                med = np.median(ts_vals)
                ax_kde.axhline(med, color=color, linestyle="--", linewidth=0.8, alpha=0.9)
            else:
                # fallback: draw histogram as horizontal bar-style KDE analog
                counts, bin_edges = np.histogram(ts_vals, bins=hist_bins, density=True)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                counts_norm = counts / counts.max() if counts.max() > 0 else counts
                ax_kde.plot(counts_norm, bin_centers, color=color, linewidth=1.6, label=label)
                med = np.median(ts_vals)
                ax_kde.axhline(med, color=color, linestyle="--", linewidth=0.8, alpha=0.9)
        except Exception as e:
            # if anything goes wrong in KDE, fallback to histogram plot and annotate
            counts, bin_edges = np.histogram(ts_vals, bins=hist_bins, density=True)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            counts_norm = counts / counts.max() if counts.max() > 0 else counts
            ax_kde.plot(counts_norm, bin_centers, color=color, linewidth=1.6, label=label + " (hist fallback)")
            med = np.median(ts_vals)
            ax_kde.axhline(med, color=color, linestyle="--", linewidth=0.8, alpha=0.9)

    # finalize time-series axis
    ax_ts.set_title(f"{antibody} — {feature} / {subfeature} (time series)")
    ax_ts.set_xlabel("timepoint")
    ax_ts.set_ylabel("value")
    ax_ts.legend(loc="upper right", fontsize="small")
    ax_ts.grid(True, alpha=0.25)

    # finalize KDE axis (note: densities plotted horizontally)
    ax_kde.set_title("Value density (KDE / hist fallback)")
    ax_kde.set_xlabel("normalized density")
    ax_kde.set_ylabel("value")
    ax_kde.legend(loc="upper right", fontsize="small")
    ax_kde.grid(True, alpha=0.25)

    plt.tight_layout()
    if show:
        plt.show()

    return fig

