import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import pickle


header = ['size',
          'frequency',
          'gdsf',
          'recency',
          'logical_recency',
          'exponential_recency',
          'exponential_logical_recency',
          'binary_recency']

subscripts = [
    'Size, Bytes',
    'Frequency',
    'GDSF, Sec/Bytes',
    'Seconds',
    'Requests',
    'Seconds',
    'Requests',
    'Indicator'
]

label = [
    'Size',
    'Frequency',
    'GDSF = Frequency / Size',
    'Recency',
    'Logical Recency',
    'Exponential Weighted Recency',
    'Exponential Weighted Logical Recency',
    'Binary Logical Recency'
]

drop_ohw_local = [
    False,
    False,
    False,
    True,
    True,
    True,
    True,
    False
]

convertors = [
    lambda x: x,
    lambda x: np.exp(-x),
    lambda x: np.exp(x),
    lambda x: 1 + x.astype(np.float64),
    lambda x: x.astype(np.float64),
    lambda x: 1 + x.astype(np.float64),
    lambda x: x.astype(np.float64),
    lambda x: 1e-10 + (x + 1) / 2.0
]


def load_file(fname, lmax=None):
    result = []
    ifile = open(fname, 'r')
    counter = 0
    for line in tqdm(ifile, total=lmax):
        if lmax is not None and counter >= lmax:
            break
        counter += 1
        result.append(np.asarray([float(item) for item in line.split(' ')]))
    return np.asarray(result)


def ecdf(sample):
    sample = np.atleast_1d(sample)
    quantiles, counts = np.unique(sample, return_counts=True)
    cumprob = np.cumsum(counts).astype(np.double) / sample.size
    return quantiles, cumprob


def plot_cum(data, label, log=True):
    x, y = ecdf(data)
    if log:
        plt.xscale('log')
    plt.plot(x, y, label=label, lw=2)


def get_percentiles(data, steps):
    result = [min(data) - 1]
    for i in range(1, steps):
        result.append(np.quantile(data, i * 1. / steps))
    result.append(max(data) + 1)
    return result


def filter_on_percetile(data, low, top):
    filter_a = data < top
    filter_b = data >= low
    return np.logical_and(filter_a, filter_b)


def sample_values(data, mval):
    print 'Sampling'
    return np.asarray([np.random.choice(range(0, mval), p=item/sum(item)) for item in tqdm(data)])


def smooth(data, factor):
    data_new = []
    for i in range(0, len(data), factor):
        data_new.append(np.mean(data[i:i + factor]))
    return data_new


def load_data(filepath, filename):
    filenames = [(join(filepath, f), int(f.replace(filename + '_', ''))) for f in listdir(filepath)
                 if isfile(join(filepath, f)) and filename in f]
    filenames = sorted(filenames, key=lambda x: x[1])
    filenames = [item[0] for item in filenames]

    time_data = []
    flow_data = []
    graph_data = {}

    for filename in filenames:
        od = pickle.load(open(filename, 'r'))
        for key in od.keys():
            if key == 'time' or key == 'flow':
                continue
            if key not in graph_data.keys():
                graph_data[key] = []
            graph_data[key] += od[key]

        time_data += od['time']
        flow_data.append(od['flow'])

    return graph_data, time_data, flow_data
