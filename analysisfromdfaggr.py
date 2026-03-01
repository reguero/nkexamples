#!/usr/bin/env python

import bioread
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
import os
import pickle

#pd.options.display.max_columns = None
#pd.options.display.max_rows = None

#for filename in ['RAW_data/PB5.acq', 'RAW_data/PB12.acq', 'RAW_data/PB13.acq', 'RAW_data/PB17.acq']:
#    file = bioread.read_file(filename)


#filename = 'RAW_data/PB17.acq'
#filename = 'RAW_data/PB5.acq'
#filename = 'RAW_data/PB12.acq'
#filename = 'RAW_data/PB13.acq'
#file = bioread.read_file(filename)
#file = bioread.read_file('RAW_data/PB12.acq')
#file = bioread.read_file('RAW_data/PB13.acq')

#print(file)
#print(file.channels)
#print(file.named_channels)

#freq_list = []
#for channel in file.named_channels:
#       freq_list.append(file.named_channels[channel].samples_per_second)
#sampling_rate = np.max(freq_list)

#data = {}
#for channel in file.named_channels:
#    print(file.named_channels[channel])
#    print(dir(file.named_channels[channel]))
#    print(file.named_channels[channel].name)
#    print(file.named_channels[channel].units)
#    print(file.named_channels[channel].data)
#    signal = np.array(file.named_channels[channel].data)
#    data[channel] = signal

# Final dataframe
#df = pd.DataFrame(data)
#print(sampling_rate)
#print(df)

#print(file.event_markers)
#sample_index = {}
#for m in file.event_markers:
#    #print(dir(m))
#    #print(str(m))
#    #print(str(m.sample_index))
#    #ooprint(str(df[m.sample_index]))
#    print(f"Marker '{m.text}' at sample {m.sample_index}")
#    #print(f"Marker '{m.text}' at time {m.time_index}")
#    #print('{0}: Channel {1}, type {2}'.format(m.text, m.channel_name, m.type))
#    #sample_index[m.text] = m.sample_index
#    #for channel in file.named_channels:
#    #   print(file.named_channels[channel].data[m.sample_index])

#data = {}
#for channel in file.named_channels:
#    signal = np.array(file.named_channels[channel].data[sample_index['start ABBA']:sample_index['end ABBA']])
#    #signal = np.array(file.named_channels[channel].data[sample_index['start ABBA']:sample_index['ABBA end']])
#    data[channel] = signal
#ABBA_df = pd.DataFrame(data)
#print(ABBA_df)

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
                                                                                           
def ECG_report(df, name, fdat, fnam):
    print('Segment {0}.{1}:'.format(fnam, name))
    #nk.signal_plot(df, subplots=True, sampling_rate=1000)
    #fig = plt.gcf()
    #fig.savefig("all_{0}.png".format(name))
    #plt.close()

    #print('ECG of segment {0}.{1}:'.format(fnam, name))
    #nk.signal_plot(df['ECG (.5 - 35 Hz)'], sampling_rate=1000)
    #fig = plt.gcf()
    #fig.savefig("ecg_{0}.png".format(name))
    #plt.close()
    # Find peaks
    #peaks, info = nk.ecg_peaks(df['ECG (.5 - 35 Hz)'], sampling_rate=1000)
    # Compute HRV indices
    #hrv = nk.hrv_time(peaks, sampling_rate=1000, show=True)
    #fig = plt.gcf()
    #fig.savefig("hrv_{0}_{1}.png".format(fnam, name))
    #plt.close(fig)
    #print('HRV of segment {0}.{1}:'.format(fnam, name))
    #print(hrv)
    #for col in hrv:
    #    print(col)
    #print(hrv['HRV_MeanNN'])
    #print(hrv.iloc[0]['HRV_MeanNN'])
    #print(hrv['HRV_SDNN'])
    #print(hrv.iloc[0]['HRV_SDNN'])
    #print(hrv['HRV_RMSSD'])
    #print(hrv.iloc[0]['HRV_RMSSD'])

    # Preprocess ECG signal
    clean_signals, info = nk.ecg_process(df['ECG (.5 - 35 Hz)'], sampling_rate=1000)
    # Visualize
    nk.ecg_plot(clean_signals, info)
    fig = plt.gcf()
    fig.savefig("ecg_proccess_{0}_{1}.png".format(fnam, name))
    plt.close()
    # Analyze
    analyze_df = nk.ecg_analyze(clean_signals, sampling_rate=1000)
    print('ECG analyze output of segment {0}.{1}:'.format(fnam, name))
    print(analyze_df)
    if name =='baseline'or (name == 'stress2_ABBA' and fnam == 'PB1_only_part2'):
        fdat.analyze_ecg = analyze_df
    else:
        #fdat.analyze_ecg = fdat.analyze_ecg.append(analyze_df)
        fdat.analyze_ecg = pd.concat([fdat.analyze_ecg, analyze_df], ignore_index=True)
    #for col in analyze_df:
    #    print(col)
    print(analyze_df['HRV_MeanNN'].apply(lambda x: np.array(x).flatten()[0]))
    #print(analyze_df['HRV_MeanNN'].iloc[0][0][0])
    print(analyze_df['HRV_SDNN'].apply(lambda x: np.array(x).flatten()[0]))
    #print(analyze_df.iloc[0]['HRV_SDNN'][0][0])
    print(analyze_df['HRV_RMSSD'].apply(lambda x: np.array(x).flatten()[0]))
    #print(analyze_df.iloc[0]['HRV_RMSSD'][0][0])

def EDA_report(df, name, fdat, fnam):
    reportname = 'EDAreport_{0}_{1}.html'.format(fnam, name)
    signals, info = nk.eda_process(df['EDA (0 - 35 Hz)'], sampling_rate=1000, report=reportname)
    #signals, info = nk.eda_process(df['EDA (0 - 35 Hz)'], sampling_rate=1000, report="text")
    #signals, info = nk.eda_process(df['EDA (0 - 35 Hz)'], sampling_rate=1000)
    nk.eda_plot(signals, info)
    fig = plt.gcf()
    fig.savefig("eda_{0}_{1}.png".format(fnam, name))
    plt.close(fig)
    print('EDA of segment {0}.{1}:'.format(fnam, name))
    #print(signals)
    #print(info)
    analyze_df = nk.eda_analyze(signals, sampling_rate=1000)
    print(analyze_df)
    if name =='baseline'or (name == 'stress2_ABBA' and fnam == 'PB1_only_part2'):
        fdat.analyze_eda = analyze_df
    else:
        #fdat.analyze_eda = fdat.analyze_eda.append(analyze_df)
        fdat.analyze_eda = pd.concat([fdat.analyze_eda, analyze_df], ignore_index=True)


def sort_filelist(l):
    import re
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split(r'PB(\d+)', key)]
    l.sort(key=alphanum)
    return l

def delta(orig, dest, df):
    ### Create a mask for the specific participant and segment
    ##mask = (df['Participant'] == 'PB26') & (df['Segment'] == 'stress1_MIST')

    ### Apply the mask to see the full rows
    ##duplicates_pb26 = df[mask]

    ##print(duplicates_pb26)

    # List of columns to analyze
    metrics = ['ECG_Rate_Mean', 'HRV_RMSSD', 'HRV_SDNN', 'HRV_MeanNN', 'EDA_Tonic_Mean', 'EDA_Tonic_SD', 'SCR_Peaks_Amplitude_Mean', 'Sympathetic_Percent', 'SCR_Frequency_PerMin', 'Unlim_Duration_Blk']

    # Pivot - this creates a DataFrame where columns are (Metric, Segment)
    pivoted_all = df.pivot(index='Participant', columns='Segment', values=metrics)
    #print(pivoted_all)

    # Subtract the entire 'orig' slice from the 'dest' slice
    # .xs (cross-section) allows us to select all metrics for one specific Segment
    df_dest = pivoted_all.xs(dest, axis=1, level='Segment')
    df_orig = pivoted_all.xs(orig, axis=1, level='Segment')
    
    deltas = df_dest - df_orig
    
    print(f"Deltas from {orig} to {dest}:")
    print(deltas)
    return deltas

def main():
    #fdatas = {}
    #labeled_files = os.listdir('RAW_data/labeled')
    #for lbfile in sort_filelist(labeled_files):
    #    if lbfile.endswith(".acq"):
    #       filename = lbfile.removesuffix(".acq")
    #       print(filename)
    #       fdata = Filedata(filename)
    #       fdata.preparedata()
    #       fdatas[filename] = fdata
    # Pickling (serializing) to a file
    #with open('dataframes.pkl', 'wb') as f:
    #    pickle.dump(fdatas, f)

    # Unpickling (deserializing) from a file
    #with open('dfmasterV0.pkl', 'rb') as f:
    with open('dfmaster.pkl', 'rb') as f:
        master_df = pickle.load(f)

    # Fix PB12 with values from B12-partie_2
    # 1. Define the segments to be replaced
    target_segments = ['stress2_MIST', 'nf2_2D']
    # 2. Extract the "Source" data (from PB12-partie_2)
    # We make a .copy() to avoid modifying the original dataframe accidentally
    source_data = master_df[
        (master_df['Participant'] == 'PB12-partie_2') &
        (master_df['Segment'].isin(target_segments))].copy()
    # 3. Change the ID in the source data to match the "Target" (PB12)
    source_data['Participant'] = 'PB12'
    # 4. Remove the "Old/Corrupt" segments from the target (PB12)
    # This prevents the 'Duplicate entries' error in your delta function
    master_df = master_df.drop(
        master_df[
            (master_df['Participant'] == 'PB12') &
            (master_df['Segment'].isin(target_segments))
        ].index
    )
    # 5. Append the "New" data from PB12-partie_2 into the master dataframe
    master_df = pd.concat([master_df, source_data], ignore_index=True)
    ## 6. Optional: Remove the temporary 'PB12-partie_2' rows if you no longer need them
    master_df = master_df[master_df['Participant'] != 'PB12-partie_2']
    print("Final segments for PB12:")
    print(master_df.query("Participant == 'PB12'")['Segment'].unique())


    print(master_df)
    print(master_df.columns)
    print(master_df.loc[:, ['ECG_Rate_Mean', 'HRV_RMSSD', 'HRV_SDNN', 'HRV_MeanNN', 'EDA_Tonic_Mean', 'EDA_Tonic_SD', 'SCR_Peaks_Amplitude_Mean', 'Sympathetic_Percent', 'SCR_Frequency_PerMin', 'Unlim_Duration_Blk', 'Participant', 'Segment']])
    ## Quick summary of all participants
    #print(master_df['EDA_Tonic_Mean'].describe())
    #print(master_df.groupby('Participant')['EDA_Tonic_Mean'])
    #print(master_df.groupby('Participant')['EDA_Tonic_Mean'].describe())
    #summary = master_df.groupby('Participant')[['EDA_Tonic_Mean', 'HRV_RMSSD', 'Sympathetic_Percent']].mean()
    #print("Average Physiological Shift:")
    #print(summary)

    # Unpickling (deserializing) from a file
    #with open('dfmasterV0.pkl', 'rb') as f:
    with open('results.pkl', 'rb') as f:
        fdatas = pickle.load(f)

    #print(master_df.groupby('Participant')['EDA_Tonic_Mean'])
    #for fnam, fdat in fdatas.items():
    #    print('\nFile {0}:'.format(fnam))
    #    for name, seg in fdat.segments.items():
    #        ECG_report(seg.df, name, fdat, fnam)
    #        EDA_report(seg.df, name, fdat, fnam)
    #        pass

    ## Pickling (serializing) to a file
    #with open('results.pkl', 'wb') as f:
    #    pickle.dump(fdatas, f)
    
    for fnam, fdat in fdatas.items():
        print('\nFile {0}:'.format(fnam))
        #print(fdat.analyze)
        #print(fdat.segments.keys())
        segnames = []
        for name, seg in fdat.segments.items():
            segnames.append(name)
            #if seg.marker_inside_index != 0:
            #    print(seg.marker_inside_index)
            #    print('segment: [{0}] before: [{1}] next [{2}] start [{3}] end [{4}] marker_inside label[{5}] marker_inside index [{6}]'.format(name, seg.before, seg.after, seg.start_index, seg.end_index, seg.marker_inside_text, seg.marker_inside_index))
            #else:
            #    print('segment: [{0}] before: [{1}] next [{2}] start [{3}] end [{4}]'.format(name, seg.before, seg.after, seg.start_index, seg.end_index))
        print(segnames)

    Group1 = ['PB2', 'PB19', 'PB23', 'PB24']
    Group2 = ['PB4', 'PB13', 'PB15', 'PB17', 'PB21']
    Group3 = ['PB3', 'PB5', 'PB22', 'PB26', 'PB27']
    Group4 = ['PB7', 'PB14', 'PB16', 'PB25', 'PB12']
    # 1. Create a mapping dictionary
    group_map = {}
    for pb in Group1: group_map[pb] = 'Group1'
    for pb in Group2: group_map[pb] = 'Group2'
    for pb in Group3: group_map[pb] = 'Group3'
    for pb in Group4: group_map[pb] = 'Group4'
    # 2. Add the column to your master_df
    master_df['Experiment_Group'] = master_df['Participant'].map(group_map)

    outcome = {}
    #outcome['Group1'] = {}
    #outcome['Group2'] = {}
    #outcome['Group3'] = {}
    #outcome['Group4'] = {}

    #for fnam in Group1:
    #    print('\nGroup1 {0}:'.format(fnam))
    #    #print(fdatas[fnam].segments.keys())
    #    fnam_df = master_df[master_df['Participant'] == fnam]
    #    outcome['Group1'+':_'+'stress1_MIST'+'_'+'nf1_VR'] = delta('stress1_MIST', 'nf1_VR', fnam_df)
    #    outcome['Group1'+':_'+'stress2_ABBA'+'_'+'nf2_2D'] = delta('stress2_ABBA', 'nf2_2D', fnam_df)
    #for fnam in Group2:
    #    print('\nGroup2 {0}:'.format(fnam))
    #    #print(fdatas[fnam].segments.keys())
    #    fnam_df = master_df[master_df['Participant'] == fnam]
    #    outcome['Group2'+':_'+'stress1_ABBA'+'_'+'nf1_2D'] = delta('stress1_ABBA', 'nf1_2D', fnam_df)
    #    outcome['Group2'+':_'+'stress2_MIST'+'_'+'nf2_VR'] = delta('stress2_MIST', 'nf2_VR', fnam_df)
    #for fnam in Group3:
    #    print('\nGroup3 {0}:'.format(fnam))
    #    #print(fdatas[fnam].segments.keys())
    #    fnam_df = master_df[master_df['Participant'] == fnam]
    #    outcome['Group3'+':_'+'stress1_MIST'+'_'+'nf1_2D'] = delta('stress1_MIST', 'nf1_2D', fnam_df)
    #    outcome['Group3'+':_'+'stress2_ABBA'+'_'+'nf2_VR'] = delta('stress2_ABBA', 'nf2_VR', fnam_df)
    #for fnam in Group4:
    #    print('\nGroup4 {0}:'.format(fnam))
    #    #print(fdatas[fnam].segments.keys())
    #    fnam_df = master_df[master_df['Participant'] == fnam]
    #    outcome['Group4'+':_'+'stress1_ABBA'+'_'+'nf1_VR'] = delta('stress1_ABBA', 'nf1_VR', fnam_df)
    #    outcome['Group4'+':_'+'stress2_MIST'+'_'+'nf2_2D'] = delta('stress2_MIST', 'nf2_2D', fnam_df)
 
    # 1. Get the deltas for everyone
    all_deltas = delta('stress1_ABBA', 'nf1_VR', master_df)
    # 2. Select just Group 4 (which includes PB12)
    # Since Participant is the index of the delta output, use .loc
    group4_deltas = all_deltas.loc[all_deltas.index.isin(Group4)]
    print(group4_deltas)
    print("Average Stress Response stress1_ABBA-nf1_VR for Group 4:")
    print(group4_deltas.mean())
    ## Quick summary by Group
    #summary = master_df.groupby(['Experiment_Group', 'Segment'])['SCR_Frequency_PerMin'].mean()
    #print(summary)
    #for o in outcome:
    #     print('\nOutcome for {0}: {1}'.format(o, outcome[o]))

    #for fnam, fdat in fdatas.items():
    #    print('\nFile {0}:'.format(fnam))
    #    segnames = []
    #    for name, seg in fdat.segments.items():
    #        segnames.append(name)
    #    print(segnames)

    ## Check what segments actually exist in the source
    #print("Available segments in PB12-partie_2:")
    #print(master_df.query("Participant == 'PB12-partie_2'")['Segment'].unique())

    ##for g, v in outcome.items():
    #for g in ['Group1', 'Group2', 'Group3', 'Group4']:
    #    for g, v in outcome[g].items():
    #        for k in v:
    #            print('\nOutcome for {0}: {1}:\n{2}'.format(g, k, v[k]))

    ## Create a mask for the specific participant and segment
    #mask = (master_df['Participant'] == 'PB26') & (master_df['Segment'] == 'stress1_MIST')

    ## Apply the mask to see the full rows
    #duplicates_pb26 = master_df[mask]

    #print(duplicates_pb26)
    #print('start = {0}'.format(fdatas['PB26'].segments['stress1_MIST'].start_index))
    #print('end = {0}'.format(fdatas['PB26'].segments['stress1_MIST'].end_index))
    #print('df = {0}'.format(len(fdatas['PB26'].segments['stress1_MIST'].df)))

    ## Identify all rows that share a Participant and Segment
    #all_dupes = master_df[master_df.duplicated(subset=['Participant', 'Segment'], keep=False)]

    ## Sort them so they appear together for easy comparison
    #print(all_dupes.sort_values(by=['Participant', 'Segment']))


    #    #fdat.analyze_ecg['seg_name'] = segnames
    #    #fdat.analyze_ecg.set_index('seg_name', inplace=True)
    #    #fdat.analyze_eda['seg_name'] = segnames
    #    #fdat.analyze_eda.set_index('seg_name', inplace=True)

    #   #print(fdat.analyze_ecg)
    #    print(fdat.analyze)
    #    print(fdat.analyze.loc[:, ['ECG_Rate_Mean', 'HRV_RMSSD', 'HRV_SDNN', 'HRV_MeanNN', 'EDA_Tonic_Mean', 'EDA_Tonic_SD', 'SCR_Peaks_Amplitude_Mean', 'Sympathetic_Percent', 'SCR_Frequency_PerMin', 'Unlim_Duration_Blk]])
    #    #print(fdat.analyze_ecg.loc[:, ['ECG_Rate_Mean', 'HRV_RMSSD']])
    #    #print(fdat.analyze_eda.loc[:, ['EDA_Tonic_Mean', 'EDA_Tonic_SD', 'SCR_Peaks_Amplitude_Mean']])

    #    #print('{0} {1}'.format(fdat.analyze_ecg.loc[:, ['ECG_Rate_Mean', 'HRV_RMSSD']], fdat.analyze_eda.loc[:, ['EDA_Tonic_SD', 'SCR_Peaks_Amplitude_Mean']]))
    #    #fdat.analyze_eda.loc[:, ['EDA_Tonic_SD', 'SCR_Peaks_Amplitude_MeanSCR_Peaks_Amplitude_Mean']]
    #    #first = ""
    #    #for name, seg in segments.items():
    #    #    if seg.before == None:
    #    #        first = name
    #    #        break
    #    #seg = segments[first]
    #    #while True:
    #    #    print(seg.name)
    #    #    if seg.after == None:
    #    #        break
    #    #    seg = segments[seg.after]
    #    #In Python 3.6+ dictionaries preserve insertion order.
    #    #sys.exit(0)
    return 0

if __name__ == "__main__":
    main()

