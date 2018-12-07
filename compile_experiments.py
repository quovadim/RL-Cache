import matplotlib.pyplot as plt
import matplotlib.dates as md
import pickle
import datetime as dt
import numpy as np

import argparse
from os import listdir
from os.path import isfile, join


def collect_data(filepath):
    filepath = filepath
    filenames = [(join(filepath, f), int(f.replace('0_', ''))) for f in listdir(filepath)
                 if isfile(join(filepath, f))]
    filenames = sorted(filenames, key=lambda x: x[1])
    filenames = [item[0] for item in filenames]

    graph_data = {}

    for filename in filenames:
        od = pickle.load(open(filename, 'r'))
        for key in od.keys():
            if key == 'time':
                continue
            if key not in graph_data.keys():
                graph_data[key] = []
            graph_data[key] += od[key]

    result_data = {}

    for key in graph_data.keys():
        result_data[key] = [100 * np.mean(graph_data[key]),
                            100 * np.percentile(graph_data[key], 75),
                            100 * np.percentile(graph_data[key], 90),
                            100 * np.percentile(graph_data[key], 95),
                            100 * np.percentile(graph_data[key], 99)]
    return result_data


def print_data(data):
    data = np.asarray(data)

    data = data.T

    mean = data[0]
    perc_75 = data[1]
    perc_90 = data[2]
    perc_95 = data[3]
    perc_99 = data[4]

    val_str = ' '.join(['{:7.3f}'] * 10)

    print '\t{:^10s}'.format('Mean'), val_str.format(*(mean.tolist()))
    print '\t{:^10s}'.format('Perc 75'), val_str.format(*(perc_75.tolist()))
    print '\t{:^10s}'.format('Perc 90'), val_str.format(*(perc_90.tolist()))
    print '\t{:^10s}'.format('Perc 95'), val_str.format(*(perc_95.tolist()))
    print '\t{:^10s}'.format('Perc 99'), val_str.format(*(perc_99.tolist()))


filepaths_hk = ['tests/M6_HK_' + str(i) + '/' for i in range(10)]

data_hk = [collect_data(item) for item in filepaths_hk]

print '\tRANDOM ML'
data_ml_random = [item['ML'] for item in data_hk]
print_data(data_ml_random)
print '\tDETERMINISTIC ML'
data_ml_random = [item['DET ML'] for item in data_hk]
print_data(data_ml_random)
print '\tLRU'
data_ml_random = [item['LRU'] for item in data_hk]
print_data(data_ml_random)
print '\tGDSF'
data_ml_random = [item['GDSF'] for item in data_hk]
print_data(data_ml_random)
