Neurokit2 examples for Marta

A full chain example could be as follows:

- producedataframes.py: Read .acq files with the RAW data, with the markers in the files, build a data staructure with dataframes for each segment that we pickle.
- producedfaggrlast5min.py: From the data staructure with dataframes for each segment, run the EDA, ECG and HRV analysis and buld a master dataframe with all the metrics that we pickle.
- analysisfromdfaggr.py: From the master dataframe do the statistical analysis with LMM models.
