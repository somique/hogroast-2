This repo contains teaching material for the ILESLA DTC 2025 statistics group project "Prediction of Antibody Developability from Molecular Dynamics Data". The files contained in this repo are to accompany the data which is held in a google drive (2x ~500MB files): https://drive.google.com/drive/folders/13_goQHv07qy-fJILSXxOxyiKtzrr5BRe?usp=sharing.

**Files**
- introduction.pdf = An abridged version of the project report associated with this data. Contains introductory text laying out the motivation for acquiring and analysing the data, relevant work in the literature, and a methods section detailing the methods used to run the molecular dynamics and the feature analyses that generated the data.
- introduction.ipynb = A short Python notebook showing how to load and perform initial analysis on the data.
- features_references.json = A json dictionary of measurement information for each main feature in the data. Specifically, keys are features with eachs' value being a list of two elements, the first is the x axis unit meaning for time series data it is time, for residue series data it is residue index etc., the second is the value with units.
- jain_features_reference.json = A json dictionary where the keys are colnames of the experimental data columns from the jain_data.csv and values are abridged strings.
- jain_data.csv = Table of experimental results data for each antibody, taken from the Jain et al. paper (https://doi.org/10.1073/pnas.1616408114). Also includes the antibody sequences used in this work.

Good luck!