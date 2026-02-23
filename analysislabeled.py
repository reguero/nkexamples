import bioread
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt

#for filename in ['RAW_data/PB5.acq', 'RAW_data/PB12.acq', 'RAW_data/PB13.acq', 'RAW_data/PB17.acq']:
#    file = bioread.read_file(filename)


filename = 'PB12-partie_2'
#filename = 'PB1_only_part2'
#filename = 'PB1 (only part2).acq'
#filename = 'PB5'
#filename = 'PB12'
#filename = 'PB16'
#filename = 'PB17'
#filename = 'PB22'
#filename = 'PB13'
#filename = 'PB21'
#filename = 'PB23'
#filename = 'PB7'
#filename = 'PB14'
#filename = 'PB4'
#filename = 'PB25'
#filename = 'PB3'
#filename = 'PB15'
#filename = 'PB19'
#filename = 'PB27'
#filename = 'PB24'
#filename = 'PB26'
#filename = 'PB2'
file = bioread.read_file('RAW_data/labeled/{0}.acq'.format(filename))
#file = bioread.read_file('RAW_data/PB12.acq')
#file = bioread.read_file('RAW_data/PB13.acq')

print(file)
print(file.channels)
print(file.named_channels)

freq_list = []
for channel in file.named_channels:
       freq_list.append(file.named_channels[channel].samples_per_second)
sampling_rate = np.max(freq_list)

data = {}
for channel in file.named_channels:
    print(file.named_channels[channel])
    print(dir(file.named_channels[channel]))
    print(file.named_channels[channel].name)
    print(file.named_channels[channel].units)
    print(file.named_channels[channel].data)
    signal = np.array(file.named_channels[channel].data)
    data[channel] = signal

# Final dataframe
df = pd.DataFrame(data)
print(sampling_rate)
print(df)

print(file.event_markers)
sample_index = {}
for m in file.event_markers:
    #print(dir(m))
    #print(str(m))
    #print(str(m.sample_index))
    #ooprint(str(df[m.sample_index]))
    print(f"Marker '{m.text}' at sample {m.sample_index}")
    #print(f"Marker '{m.text}' at time {m.time_index}")
    #print('{0}: Channel {1}, type {2}'.format(m.text, m.channel_name, m.type))
    #sample_index[m.text] = m.sample_index
    #for channel in file.named_channels:
    #   print(file.named_channels[channel].data[m.sample_index])

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

    def make_df(self):
        data = {}
        for channel in file.named_channels:
            signal = np.array(file.named_channels[channel].data[self.start_index:self.end_index])
            data[channel] = signal
        self.df = pd.DataFrame(data)
        print('Dataframe of {0}:'.format(self.name))
        #print(self.df)

segments = {}
name_seg_before = None
name_seg_after = None
for m in file.event_markers:
    if m.text == 'post nf1_VR (crashed)' and filename == 'PB1_only_part2':
        continue

    if m.text == 'recording interrupted' and filename == 'PB12':
        m.text = 'stress2_MIST_end'

    if m.text == 'stress1_ABBA_start' and m.sample_index == 946410 and filename == 'PB13':
        m.text = "stress1_ABBA_end"

    if m.text == 'stress2_MIST_end' and filename == 'PB23':
        m.text = "stress1_MIST_end"

    if m.text == '' and filename == 'PB27':
        m.text = "nf2_VR_start"

    if m.text == '' and filename == 'PB15':
        m.text = "nf2_VR_end"

    if m.text == '' and filename in ['PB4', 'PB16', 'PB22', 'PB25', 'PB12-partie_2']:
        continue

    if m.text == 'baseline':
        name = m.text
        current_seg = Segment(name, m.text, m.sample_index, name_seg_before)
        segments[name] = current_seg
        name_seg_before = name
    elif m.text.startswith("Segment "):
        continue
    elif m.text.endswith("_start"):
        name = m.text.removesuffix("_start")
        current_seg = Segment(name, m.text, m.sample_index, name_seg_before)
        segments[name] = current_seg
        if name_seg_before != None:
            segments[name_seg_before].after = name
        name_seg_before = name
    elif m.text.endswith("_end"):
        name = m.text.removesuffix("_end")
        current_seg = segments[name]
        current_seg.end_index_text = m.text
        current_seg.end_index = m.sample_index
        current_seg.make_df()
        name_seg_before = name
    elif m.text.endswith("_unlimited"):
        if name_seg_before != None:
            print("unlimited: " + m.text)
            segments[name_seg_before].marker_inside_text = m.text
            segments[name_seg_before].marker_inside_index = m.sample_index
    else:
        print("unknown: " + m.text)
    print(m.text)

if filename in ['PB4', 'PB7', 'PB12', 'PB13', 'PB14', 'PB15', 'PB16', 'PB17', 'PB21', 'PB25']:
    current_seg = segments['baseline']
    current_seg.end_index_text = segments['stress1_ABBA'].start_index_text
    current_seg.end_index = segments['stress1_ABBA'].start_index
    current_seg.make_df()
    #if filename == 'PB2' or filename == 'PB26' or filename == 'PB24':
if filename in ['PB2', 'PB3', 'PB5', 'PB19', 'PB22', 'PB23', 'PB24', 'PB26', 'PB27']:
    current_seg = segments['baseline']
    current_seg.end_index_text = segments['stress1_MIST'].start_index_text
    current_seg.end_index = segments['stress1_MIST'].start_index
    current_seg.make_df()

if filename in ['PB12-partie_2']:
    current_seg = segments['baseline']
    current_seg.end_index_text = segments['stress2_MIST'].start_index_text
    current_seg.end_index = segments['stress2_MIST'].start_index
    current_seg.make_df()

for name, seg in segments.items():
        print('Segment {0}:'.format(name))
        nk.signal_plot(seg.df, subplots=True, sampling_rate=1000)
        #fig = plt.gcf()
        #fig.savefig("all_{0}.png".format(name))

        print('ECG of segment {0}:'.format(name))
        nk.signal_plot(seg.df['ECG (.5 - 35 Hz)'], sampling_rate=1000)
        #fig = plt.gcf()
        #fig.savefig("ecg_{0}.png".format(name))
        # Find peaks
        peaks, info = nk.ecg_peaks(seg.df['ECG (.5 - 35 Hz)'], sampling_rate=1000)
        # Compute HRV indices
        hrv = nk.hrv_time(peaks, sampling_rate=1000, show=True)
        fig = plt.gcf()
        fig.savefig("hrv_{0}.png".format(name))
        print('HRV of segment {0}:'.format(name))
        print(hrv)
        #for col in hrv:
        #    print(col)
        print(hrv['HRV_MeanNN'])
        #print(hrv.iloc[0]['HRV_MeanNN'])
        print(hrv['HRV_SDNN'])
        #print(hrv.iloc[0]['HRV_SDNN'])
        print(hrv['HRV_RMSSD'])
        #print(hrv.iloc[0]['HRV_RMSSD'])

        reportname = 'EDAreport_{0}.html'.format(name)
        signals, info = nk.eda_process(seg.df['EDA (0 - 35 Hz)'], sampling_rate=1000, report=reportname)
        nk.eda_plot(signals, info)
        fig = plt.gcf()
        fig.savefig("eda_{0}.png".format(name))
        print('EDA of segment {0}:'.format(name))
        #print(signals)
        #print(info)
        analyze_df = nk.eda_analyze(signals, sampling_rate=1000)
        print(analyze_df)


