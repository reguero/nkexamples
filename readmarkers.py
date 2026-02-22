import bioread
#data = bioread.read_file('RAW_data/PB5.acq')
data = bioread.read_file('RAW_data/PB17.acq')
#data = bioread.read_file('RAW_data/PB12.acq')
#data = bioread.read_file('RAW_data/PB13.acq')

print(data)

#print(data.event_markers)

for m in data.event_markers:
    print(str(m))
