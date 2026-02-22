import bioread
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt

#for filename in ['RAW_data/PB5.acq', 'RAW_data/PB12.acq', 'RAW_data/PB13.acq', 'RAW_data/PB17.acq']:
#    file = bioread.read_file(filename)


#filename = 'RAW_data/PB17.acq'
#filename = 'RAW_data/PB5.acq'
#filename = 'RAW_data/PB12.acq'
filename = 'RAW_data/PB13.acq'
file = bioread.read_file(filename)
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
    if filename == 'RAW_data/PB13.acq':
        if m.text == 'end 2D NF ':
           m.text = 'end 2D NF'
        if m.text == 'Segment 1':
            name = m.text
            current_seg = Segment(name, m.text, m.sample_index, name_seg_before)
            segments[name] = current_seg
            name_seg_before = name
    if filename == 'RAW_data/PB12.acq':
        if m.text == 'ABBA end':
           m.text = 'end ABBA'
        if m.text == 'end VR NF':
           m.text = 'end NF VR'
        if m.text == 'Segment 3':
           m.text = 'end MIST'
        if m.text == 'Segment 1' or m.text == 'Segment 2':
            name = m.text
            current_seg = Segment(name, m.text, m.sample_index, name_seg_before)
            segments[name] = current_seg
            name_seg_before = name
    if filename == 'RAW_data/PB5.acq':
        if m.text == 'end NF VR':
           m.text = 'end VR NF'
        if m.text == 'Segment 1' or m.text == 'Segment 2':
            name = m.text
            current_seg = Segment(name, m.text, m.sample_index, name_seg_before)
            segments[name] = current_seg
            name_seg_before = name
    if filename == 'RAW_data/PB17.acq':
        if m.text == 'end 2D NF' and m.sample_index == 4400451:
           m.text = 'end NF-VR'
        if m.text == 'Segment 1':
            name = m.text
            current_seg = Segment(name, m.text, m.sample_index, name_seg_before)
            segments[name] = current_seg
            name_seg_before = name
    if   m.text.startswith("start "):
        name = m.text.removeprefix("start ")
        current_seg = Segment(name, m.text, m.sample_index, name_seg_before)
        segments[name] = current_seg
        if name_seg_before != None:
            current_seg.before = name_seg_before
            segments[name_seg_before].after = name
        name_seg_before = name
    elif m.text.startswith("end "):
        name = m.text.removeprefix("end ")
        current_seg = segments[name]
        current_seg.end_index_text = m.text
        current_seg.end_index = m.sample_index
        current_seg.make_df()
        name_seg_before = name
    else:
        if name_seg_before != None:
            print("insider: " + m.text)
            segments[name_seg_before].marker_inside_text = m.text
            segments[name_seg_before].marker_inside_index = m.sample_index
    print(m.text)

if filename == 'RAW_data/PB17.acq':
    current_seg = segments['Segment 1']
    current_seg.end_index_text = segments['ABBA'].start_index_text
    current_seg.end_index = segments['ABBA'].start_index
    current_seg.make_df()
if filename == 'RAW_data/PB5.acq':
    current_seg = segments['Segment 1']
    current_seg.end_index_text = segments['Segment 2'].start_index_text
    current_seg.end_index = segments['Segment 2'].start_index
    current_seg.make_df()
    current_seg = segments['Segment 2']
    current_seg.end_index_text = segments['MIST'].start_index_text
    current_seg.end_index = segments['MIST'].start_index
    current_seg.make_df()
if filename == 'RAW_data/PB12.acq':
    current_seg = segments['Segment 1']
    current_seg.end_index_text = segments['Segment 2'].start_index_text
    current_seg.end_index = segments['Segment 2'].start_index
    current_seg.make_df()
    current_seg = segments['Segment 2']
    current_seg.end_index_text = segments['ABBA'].start_index_text
    current_seg.end_index = segments['ABBA'].start_index
    current_seg.make_df()
if filename == 'RAW_data/PB13.acq':
    current_seg = segments['Segment 1']
    current_seg.end_index_text = segments['ABBA'].start_index_text
    current_seg.end_index = segments['ABBA'].start_index
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

        #reportname = 'EDAreport_{0}.html'.format(name)
        #signals, info = nk.eda_process(seg.df['EDA (0 - 35 Hz)'], sampling_rate=1000, report=reportname)
        signals, info = nk.eda_process(seg.df['EDA (0 - 35 Hz)'], sampling_rate=1000, report="text")
        nk.eda_plot(signals, info)
        fig = plt.gcf()
        fig.savefig("eda_{0}.png".format(name))
        print('EDA of segment {0}:'.format(name))
        #print(signals)
        #print(info)
        analyze_df = nk.eda_analyze(signals, sampling_rate=1000)
        print(analyze_df)

for name, seg in segments.items():
    print('segment: [{0}] before: [{1}] after [{2}]'.format(name, seg.before, seg.after))

#first = ""
#for name, seg in segments.items():
#    if seg.before == None:
#        first = name
#        break
#seg = segments[first]
#while True:
#    print(seg.name)
#    if seg.after == None:
#        break
#    seg = segments[seg.after]
#In Python 3.6+ dictionaries preserve insertion order.
