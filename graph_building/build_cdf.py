import matplotlib.pyplot as plt
import matplotlib.dates as md
import pickle
import datetime as dt
import numpy as np

import argparse
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from graphs_auxiliary import plot_cum, smooth, load_data

parser = argparse.ArgumentParser(description='Algorithm tester')
parser.add_argument("filepath", type=str, help="Output filename")
parser.add_argument("-f", "--filename", type=str, default='', help="Output filename")
parser.add_argument('-s', '--skip', type=int, default=0, help="Skip")
parser.add_argument('-m', '--smooth', type=int, default=1, help="Smooth factor")

args = parser.parse_args()

filepath = args.filepath
filenames = [(join(filepath, f), int(f.replace(args.filename + '_', ''))) for f in listdir(filepath)
             if isfile(join(filepath, f)) and args.filename in f]
filenames = sorted(filenames, key=lambda x: x[1])
filenames = [item[0] for item in filenames]

graph_data, time_data, flow_data = load_data(args.filepath, args.filename)

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
time_data = [int(item) for item in smooth(time_data, args.smooth)]
flow_data = smooth(flow_data, args.smooth)

tgd = {}
for key in graph_data.keys():
    tgd[key] = smooth(graph_data[key][args.skip:], args.smooth)

graph_data = tgd

keys_ml = [item for item in graph_data.keys() if 'ML' in item]
keys_cl = [item for item in graph_data.keys() if 'ML' not in item]

diffs = {}
for key in keys_ml:
    diffs[key] = {}
    for key_cl in keys_cl:
        diffs[key][key_cl] = graph_data[key] - graph_data[key_cl]

for key in diffs.keys():
    for key_cl in diffs[key].keys():
        pstr = '{:^20s} 1% - {:7.4f}% 5% - {:7.4f}% 10% - {:7.4f}% 25% - {:7.4f}% 50% - {:7.4f}% 75% - {:7.4f}% 90% - {:7.4f}% 95% - {:7.4f}% 99% - {:7.4f}%'
        print pstr.format(key + ' ' + key_cl,
                      100 * np.percentile(diffs[key][key_cl], 1),
                      100 * np.percentile(diffs[key][key_cl], 5),
                      100 * np.percentile(diffs[key][key_cl], 10),
                      100 * np.percentile(diffs[key][key_cl], 25),
                      100 * np.percentile(diffs[key][key_cl], 50),
                      100 * np.percentile(diffs[key][key_cl], 75),
                      100 * np.percentile(diffs[key][key_cl], 90),
                      100 * np.percentile(diffs[key][key_cl], 95),
                      100 * np.percentile(diffs[key][key_cl], 99))

for key in diffs.keys():
    for key_cl in diffs[key].keys():
        data = 100 * np.asarray(diffs[key][key_cl])
        plot_cum(data, key + ' ' + key_cl, log=False)

#for key in graph_data.keys():
#    data = 100 * np.asarray(graph_data[key])
#    plot_cum(data, key, log=False)

#plt.xlabel('Hit Rate')
#plt.ylabel('Fraction %')

plt.legend()
plt.show()