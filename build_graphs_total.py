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
flow_data = []
graph_data = {}

for filename in filenames:
    od = pickle.load(open(filename, 'r'))
    for key in od.keys():
        if key == 'time' or key == 'flow' or key == 'size':
            continue
        if key not in graph_data.keys():
            graph_data[key] = []
        graph_data[key] += od[key]

    time_data += od['time']
    flow_data.append(od['flow'])

flow_data = flow_data[args.skip:]
#m_flow = max(flow_data)
flow_data = [item / (1024 * 1024 * 1024) for item in flow_data]

#add_key = 'AdaptSize'
#adapt_data = [int(item) for item in tqdm(open('tests/history_adapt', 'r').readlines())]
#period = 300000
#hits = 0
#misses = 0
#ah = []
#for i in tqdm(range(len(adapt_data))):
#    hits += adapt_data[i]
#    misses += 1 - adapt_data[i]
#    if i % period == 0:
#        ah.append(float(hits) / (hits + misses))
#        hits = 0
#        misses = 0
#print ah
#exit(0)
time_data = time_data[args.skip:]

tgd = {}
for key in graph_data.keys():
    tgd[key] = graph_data[key][args.skip:]

graph_data = tgd

for key in graph_data.keys():
    pstr = '{:^13s} 1% - {:7.4f}% 5% - {:7.4f}% 10% - {:7.4f}% 25% - {:7.4f}% 50% - {:7.4f}% 75% - {:7.4f}% 90% - {:7.4f}% 95% - {:7.4f}% 99% - {:7.4f}%'
    print pstr.format(key,
                      100 * np.percentile(graph_data[key], 1),
                      100 * np.percentile(graph_data[key], 5),
                      100 * np.percentile(graph_data[key], 10),
                      100 * np.percentile(graph_data[key], 25),
                      100 * np.percentile(graph_data[key], 50),
                      100 * np.percentile(graph_data[key], 75),
                      100 * np.percentile(graph_data[key], 90),
                      100 * np.percentile(graph_data[key], 95),
                      100 * np.percentile(graph_data[key], 99))


fig, ax = plt.subplots()

accumulated_time = [dt.datetime.fromtimestamp(ts) for ts in time_data]
xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
ax.xaxis.set_major_formatter(xfmt)

for key in graph_data.keys():
    if key == 'time':
        continue
    data = 100 * np.asarray(graph_data[key])
    ax.plot(accumulated_time, data, label=key)

ax2 = ax.twinx()
ax2.xaxis.set_major_formatter(xfmt)
ax2.plot(accumulated_time, flow_data, label='Flow', lw=7, alpha=0.4)
#ax2.set_ylabel('GiB per second')

fig.autofmt_xdate()

ax.set_xlabel('Time YYYY-MM-DD HH-MM-SS')
ax.set_ylabel('Hit Rate %')

ax.legend()
plt.show()
