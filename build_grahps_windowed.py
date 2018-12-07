import matplotlib.pyplot as plt
import matplotlib.dates as md
import pickle
import datetime as dt
import numpy as np

import argparse
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Algorithm tester')
parser.add_argument("filepath", type=str, help="Output filename")
parser.add_argument("-f", "--filename", type=str, default='', help="Output filename")
parser.add_argument('-s', '--skip', type=int, default=100000, help="Skip")

args = parser.parse_args()

filepath = args.filepath
filenames = [(join(filepath, f), int(f.replace(args.filename + '_', ''))) for f in listdir(filepath)
             if isfile(join(filepath, f)) and args.filename in f]
filenames = sorted(filenames, key=lambda x: x[1])
filenames = [item[0] for item in filenames]

time_data = []
graph_data = {}


def convert(data, prev_value):
    data = [prev_value] + data
    ndata = []
    for i in range(1, len(data)):
        if data[i-1] < data[i]:
            ndata.append(1)
        else:
            ndata.append(0)
    return ndata, data[len(data) - 1]


def windowize(data, time, window_size):
    window = []
    time_window = []
    result = []
    for i in tqdm(range(len(data))):
        window.append(data[i])
        time_window.append(time[i])
        while time[i] - time_window[0] > window_size:
            window = window[1:]
            time_window = time_window[1:]
        result.append(sum(window) * 1.0 / len(window))
    return result


prev_values = {}

for filename in filenames:
    od = pickle.load(open(filename, 'r'))
    for key in od.keys():
        if key == 'time':
            continue
        if key not in graph_data.keys():
            graph_data[key] = []
        d, v = convert(od[key], prev_values.get(key, 0))
        prev_values[key] = v
        graph_data[key] += d

    time_data += od['time']

wdata = {}
for key in graph_data.keys():
    wdata[key] = windowize(graph_data[key], time_data, 60)
graph_data = wdata

print len(time_data)

time_data = time_data[args.skip:]

tgd = {}
for key in graph_data.keys():
    tgd[key] = graph_data[key][args.skip:]

graph_data = tgd

fig, ax = plt.subplots()

accumulated_time = [dt.datetime.fromtimestamp(ts) for ts in time_data]
xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
ax.xaxis.set_major_formatter(xfmt)

for key in graph_data.keys():
    if key == 'time':
        continue
    data = 100 * np.asarray(graph_data[key])
    ax.plot(accumulated_time, data, label=key)

fig.autofmt_xdate()

plt.xlabel('Time YYYY-MM-DD HH-MM-SS')
plt.ylabel('Hit Rate %')

plt.legend()
plt.show()
