#!/usr/bin/env python

import bioread
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
import os
import pickle

class Segment(object):
    def __init__(self, name, start_index_text, start_index, seg_before):
        self.name = name
        self.start_index_text = start_index_text
        self.start_index = start_index
        self.end_index_text = ""
        self.end_index = 0
        self.before = seg_before
        self.after = None
        self.marker_inside_text = ""
        self.marker_inside_index = 0

    def make_df(self, file):
        data = {}
        for channel in file.named_channels:
            signal = np.array(file.named_channels[channel].data[self.start_index:self.end_index])
            data[channel] = signal
        self.df = pd.DataFrame(data)
        print('Dataframe of {0}:'.format(self.name))
        #print(self.df)

class Filedata(object):
    def __init__(self, filename):
        self.filename = filename
        self.segments = {}
        #self.analyze_ecg = {}
        #self.analyze_eda = {}
        #self.analyze = {}

    def preparedata(self):
        name_seg_before = None
        name_seg_after = None
        file = bioread.read_file('RAW_data/labeled/{0}.acq'.format(self.filename))
        for m in file.event_markers:
            if m.text == 'post nf1_VR (crashed)' and self.filename == 'PB1_only_part2':
                continue

            if m.text == 'recording interrupted' and self.filename == 'PB12':
                m.text = 'stress2_MIST_end'

            if m.text == 'stress1_ABBA_start' and m.sample_index == 946410 and self.filename == 'PB13':
                m.text = "stress1_ABBA_end"

            if m.text == 'stress2_MIST_end' and self.filename == 'PB23':
                m.text = "stress1_MIST_end"

            if m.text == '' and self.filename == 'PB27':
                m.text = "nf2_VR_start"

            if m.text == '' and self.filename == 'PB15':
                m.text = "nf2_VR_end"

            if m.text == '' and self.filename in ['PB4', 'PB16', 'PB22', 'PB25', 'PB12-partie_2']:
                continue

            if m.text == 'baseline':
                name = m.text
                current_seg = Segment(name, m.text, m.sample_index, name_seg_before)
                self.segments[name] = current_seg
                name_seg_before = name
            elif m.text.startswith("Segment "):
                continue
            elif m.text.endswith("_start"):
                name = m.text.removesuffix("_start")
                current_seg = Segment(name, m.text, m.sample_index, name_seg_before)
                self.segments[name] = current_seg
                if name_seg_before != None:
                    self.segments[name_seg_before].after = name
                name_seg_before = name
            elif m.text.endswith("_end"):
                name = m.text.removesuffix("_end")
                current_seg = self.segments[name]
                current_seg.end_index_text = m.text
                current_seg.end_index = m.sample_index
                current_seg.make_df(file)
                name_seg_before = name
            elif m.text.endswith("_unlimited"):
                if name_seg_before != None:
                    print("unlimited: " + m.text)
                    self.segments[name_seg_before].marker_inside_text = m.text
                    self.segments[name_seg_before].marker_inside_index = m.sample_index
            else:
                print("unknown: " + m.text)
            print(m.text)

        if self.filename in ['PB4', 'PB7', 'PB12', 'PB13', 'PB14', 'PB15', 'PB16', 'PB17', 'PB21', 'PB25']:
            current_seg = self.segments['baseline']
            current_seg.end_index_text = self.segments['stress1_ABBA'].start_index_text
            current_seg.end_index = self.segments['stress1_ABBA'].start_index
            current_seg.make_df(file)

        if self.filename in ['PB2', 'PB3', 'PB5', 'PB19', 'PB22', 'PB23', 'PB24', 'PB26', 'PB27']:
            current_seg = self.segments['baseline']
            current_seg.end_index_text = self.segments['stress1_MIST'].start_index_text
            current_seg.end_index = self.segments['stress1_MIST'].start_index
            current_seg.make_df(file)

        if self.filename in ['PB12-partie_2']:
            current_seg = self.segments['baseline']
            current_seg.end_index_text = self.segments['stress2_MIST'].start_index_text
            current_seg.end_index = self.segments['stress2_MIST'].start_index
            current_seg.make_df(file)
                                                                                           
def ECG_report(seg, name, fdat, fnam):
    print('Segment {0}.{1}:'.format(fnam, name))
    # Preprocess ECG signal
    clean_signals, info = nk.ecg_process(seg.df['ECG (.5 - 35 Hz)'], sampling_rate=1000)
    # Visualize
    nk.ecg_plot(clean_signals, info)
    fig = plt.gcf()
    fig.savefig("ecg_proccess_{0}_{1}.png".format(fnam, name))
    plt.close()
    # Analyze
    analyze_df = nk.ecg_analyze(clean_signals, sampling_rate=1000)
    # Flatten the columns if they are nested lists
    for col in analyze_df.columns:
        analyze_df[col] = analyze_df[col].apply(lambda x: x[0][0] if isinstance(x, (list, np.ndarray)) else x)
    print('ECG analyze output of segment {0}.{1}:'.format(fnam, name))
    print(analyze_df)
    fdat.analyze_ecg = analyze_df
    #print(analyze_df['HRV_MeanNN'].apply(lambda x: np.array(x).flatten()[0]))
    ##print(analyze_df['HRV_MeanNN'].iloc[0][0][0])
    #print(analyze_df['HRV_SDNN'].apply(lambda x: np.array(x).flatten()[0]))
    ##print(analyze_df.iloc[0]['HRV_SDNN'][0][0])
    #print(analyze_df['HRV_RMSSD'].apply(lambda x: np.array(x).flatten()[0]))
    ##print(analyze_df.iloc[0]['HRV_RMSSD'][0][0])

def EDA_report(seg, name, fdat, fnam):
    reportname = 'EDAreport_{0}_{1}.html'.format(fnam, name)
    signals, info = nk.eda_process(seg.df['EDA (0 - 35 Hz)'], sampling_rate=1000, report=reportname)
    #signals, info = nk.eda_process(seg.df['EDA (0 - 35 Hz)'], sampling_rate=1000, report="text")
    #signals, info = nk.eda_process(seg.df['EDA (0 - 35 Hz)'], sampling_rate=1000)
    nk.eda_plot(signals, info)
    fig = plt.gcf()
    fig.savefig("eda_{0}_{1}.png".format(fnam, name))
    plt.close(fig)
    print('EDA of segment {0}.{1}:'.format(fnam, name))
    #print(signals)
    #print(info)
    analyze_df = nk.eda_analyze(signals, sampling_rate=1000)
    # Manually add the average Tonic level (SCL) for this segment
    analyze_df['EDA_Tonic_Mean'] = signals['EDA_Tonic'].mean()
    # Calculate duration in minutes
    duration_min = len(seg.df) / 1000 / 60
    print('duration_min = {0}'.format(duration_min))
    # Add Frequency to your analyze_df
    analyze_df['SCR_Frequency_PerMin'] = analyze_df['SCR_Peaks_N'] / duration_min
    # Scale the Sympathetic index to be more readable (Percentage)
    analyze_df['Sympathetic_Percent'] = analyze_df['EDA_SympatheticN'] * 100
    print(analyze_df)
    fdat.analyze_eda = analyze_df

def Unlimited_duration_block(seg, name, fdat, fnam):
    unlimited_duration_sec = 0
    if seg.marker_inside_index != 0:
        unlimited_duration_sec = (seg.end_index - seg.marker_inside_index)/1000
    print('time in unlimited duration block = {0}'.format(unlimited_duration_sec))
    return unlimited_duration_sec

def sort_filelist(l):
    import re
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split(r'PB(\d+)', key)]
    l.sort(key=alphanum)
    return l

def main():

    # Unpickling (deserializing) from a file
    with open('dataframes.pkl', 'rb') as f:
        fdatas = pickle.load(f)

    master_data = []
    for fnam, fdat in fdatas.items():
        print('\nFile {0}:'.format(fnam))
        for name, seg in fdat.segments.items():
            ECG_report(seg, name, fdat, fnam)
            EDA_report(seg, name, fdat, fnam)
            # Merge DFs
            fdat.analyze = pd.concat([fdat.analyze_eda, fdat.analyze_ecg], axis=1)
            # Drop duplicated columns
            fdat.analyze = fdat.analyze.loc[:, ~fdat.analyze.columns.duplicated()]
            # Add Segment Label
            fdat.analyze['Segment'] = name
            #Add participant
            fdat.analyze['Participant'] = fnam
            # Add time in limited duration block
            fdat.analyze['Unlim_Duration_Blk'] = Unlimited_duration_block(seg, name, fdat, fnam)
            master_data.append(fdat.analyze)
            #fdat.analyze.set_index('Segment_Label', inplace=True)
        print(fdat.analyze)
        # Drop original Dfs
        del fdat.analyze_ecg
        del fdat.analyze_eda

    # Create the Master Dataframe
    df_master = pd.concat(master_data, ignore_index=True)
    # Pickling (serializing) to a file
    with open('dfmaster.pkl', 'wb') as f:
        pickle.dump(df_master, f)

    # Pickling (serializing) to a file
    with open('results.pkl', 'wb') as f:
        pickle.dump(fdatas, f)

        print('\nFor file {0}:'.format(fnam))
        # Please note that in Python 3.6+ dictionaries preserve insertion order.
        for name, seg in fdat.segments.items():
            if seg.marker_inside_index != 0:
                print(seg.marker_inside_index)
                print('segment: [{0}] before: [{1}] after [{2}] start [{3}] end [{4}] marker_inside label[{5}] marker_inside index [{6}]'.format(name, seg.before, seg.after, seg.start_index, seg.end_index, seg.marker_inside_text, seg.marker_inside_index))
            else:
                print('segment: [{0}] before: [{1}] after [{2}] start [{3}] end [{4}]'.format(name, seg.before, seg.after, seg.start_index, seg.end_index))
    return 0

if __name__ == "__main__":
    main()

