import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime as dt
import numpy as np

import argparse

from graphs_auxiliary import smooth, load_data

parser = argparse.ArgumentParser(description='Algorithm tester')
parser.add_argument("-f", "--filename", type=str, default='', help="Output filename")
parser.add_argument('-s', '--skip', type=int, default=0, help="Skip")
parser.add_argument('-m', '--smooth', type=int, default=1, help="Smooth factor")

args = parser.parse_args()

ML_filepath = 'tests/ml_only_usa/'
GD_filepath = 'tests/small_hk_2048/'
graph_data, time_data, flow_data = load_data(ML_filepath, args.filename)
data = {}
maxl = None
for key in graph_data:
    if 'ML' not in key:
        continue
    splitted = key.split('-')
    size = int(splitted[3])
    rng = splitted[2]
    if size not in data.keys():
        data[size] = {}

    data[size][rng] = graph_data[key]
    if maxl is None or len(graph_data[key]) < maxl:
        maxl = len(graph_data[key])

graph_data, time_data, flow_data = load_data(GD_filepath, args.filename)

for key in graph_data:
    if 'ML' in key:
        continue

    splitted = key.split('-')
    size = int(splitted[2])
    rng = splitted[0]
    if size not in data.keys():
        data[size] = {}

    data[size][rng] = graph_data[key]
    if maxl is None or len(graph_data[key]) < maxl:
        maxl = len(graph_data)

names = ['mean',
         '1% percentile',
         '5% percentile',
         '10% percentile',
         '25% percentile',
         '50% percentile',
         '75% percentile',
         '90% percentile',
         '95% percentile',
         '99% percentile']


def get_stats(data):
    return [100 * np.mean(data),
            100 * np.percentile(data, 1),
            100 * np.percentile(data, 5),
            100 * np.percentile(data, 10),
            100 * np.percentile(data, 25),
            100 * np.percentile(data, 50),
            100 * np.percentile(data, 75),
            100 * np.percentile(data, 90),
            100 * np.percentile(data, 95),
            100 * np.percentile(data, 99)]


data_mod = {}
for key in data.keys():
    data_mod[key] = {}
    for key_l in data[key].keys():
        data_mod[key][key_l] = get_stats(data[key][key_l][args.skip:maxl])

graphs = {}

for name in names:
    index = names.index(name)
    x = data_mod.keys()
    y = {}
    for key in data_mod.keys():
        for key_l in data_mod[key].keys():
            if key_l not in y.keys():
                y[key_l] = []
            y[key_l].append(data_mod[key][key_l][index])

    graphs[name] = (x, y)

for key in graphs.keys():
    plt.clf()
    x, y = graphs[key]
    plt.title(key)
    for lkey in y.keys():
        data_zipped = sorted(zip(x, y[lkey]), key=lambda x: x[0])
        x_l, y_l = zip(*data_zipped)
        plt.plot([str(item/1024) + 'GB' for item in x_l], y_l, label=lkey)
    plt.legend()
    plt.xlabel('Size')
    plt.ylabel('Hit Rate')
    plt.savefig(key.replace(' ', '_').replace('%', '') + '.png')


for lkey in graphs['mean'][1].keys():
    plt.clf()
    plt.title(lkey)
    for key in ['25% percentile', '50% percentile', '75% percentile']:
        x, y = graphs[key]
        data_zipped = sorted(zip(x, y[lkey]), key=lambda x: x[0])
        x_l, y_l = zip(*data_zipped)
        plt.plot([str(item/1024) + 'GB' for item in x_l], y_l, label=key + ' ' + lkey)
    plt.legend()
    plt.xlabel('Size')
    plt.ylabel('Hit Rate')
    plt.savefig(lkey + 'combined.png')

