import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import pickle

import matplotlib.dates as md
import datetime as dt
from hurry.filesize import size as hurry_fsize
from environment.environment_aux import parse_name


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

statistics_names = [
    'mean',
    'std',
    '1% percentile',
    '5% percentile',
    '10% percentile',
    '25% percentile',
    '50% percentile',
    '75% percentile',
    '90% percentile',
    '95% percentile',
    '99% percentile'
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


def get_graph_label(key):
    kdata = key.split('-')
    fdata = []
    size = hurry_fsize(int(kdata[3]) * 1024 * 1024)
    label_adm = kdata[0]
    if label_adm == 'AL':
        label_adm = 'ALL'
    fdata.append(label_adm)
    label_evc = kdata[1]
    fdata.append(label_evc)
    if label_adm == 'ML' or label_evc == 'ML':
        fdata.append(kdata[2][0])
    fdata.append(size)
    return ' '.join(fdata)


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
        data_new.append(np.mean(data[i:min(i + factor, len(data))]))
    return data_new


def get_number_of_steps(filepath, filename, skip):
    file_names = [(join(filepath, f), int(f.replace(filename + '_', ''))) for f in listdir(filepath)
                 if isfile(join(filepath, f)) and filename in f]
    file_names = sorted(file_names, key=lambda x: x[1])
    file_names = [item[0] for item in file_names]

    return len(file_names) - skip


def load_data(filepath, filename, skip, max_length=None):
    file_names = [(join(filepath, f), int(f.replace(filename + '_', ''))) for f in listdir(filepath)
                 if isfile(join(filepath, f)) and filename in f]
    file_names = sorted(file_names, key=lambda x: x[1])
    file_names = [item[0] for item in file_names]

    time_data = []
    flow_data = []
    alphas = None
    graph_data = {}

    if max_length is not None:
        file_names = file_names[:max_length]

    for filename in file_names[skip:]:
        od = pickle.load(open(filename, 'r'))
        if alphas is None:
            alphas = od['alphas']
        for key in od.keys():
            if key == 'time' or key == 'flow' or key == 'alphas':
                continue
            if key not in graph_data.keys():
                graph_data[key] = []
            graph_data[key] += od[key]

        time_data += od['time']
        flow_data.append(od['flow'])

    if alphas is None:
        alphas = [-1.]

    alphas_mapping = {-1.: 'UNKNOWN', 1.: 'BHR', 0.5: 'EQMix', 0.: 'OHR'}
    alphas_text = []
    for alpha in alphas:
        if alpha in alphas_mapping.keys():
            alphas_text.append(alphas_mapping[alpha])
        else:
            alphas_text.append('UN' + str(alpha))
    graph_data_transposed = {}
    for key in graph_data.keys():
        data = graph_data[key]
        alphas_data = {}
        for i in range(len(alphas_text)):
            alphas_data[alphas_text[i]] = [item[i] for item in data]
        graph_data_transposed[key] = alphas_data
    graph_data = graph_data_transposed

    return graph_data, time_data, flow_data, alphas_text


def load_dataset(folder, filename, skip, keys_to_ignore, max_length=None):
    graph_data, time_data, flow_data, alphas = load_data(folder, filename, skip, max_length)
    time_data = [int(item) for item in time_data]
    flow_data = flow_data

    keys = [key for key in graph_data.keys() if key not in keys_to_ignore]
    for key in keys_to_ignore:
        del graph_data[key]

    names_info = {}
    for key in keys:
        names_info[key] = parse_name(key)

    print '...Done'
    statistics = {}
    for key in keys:
        statistics[key] = {}
        for alpha in graph_data[key].keys():
            statistics[key][alpha] = get_stats(graph_data[key][alpha])

    return graph_data, time_data, flow_data, alphas, names_info, statistics


def get_stats(data_vector):
    statistics_vector = [100 * np.mean(data_vector),
                         100 * np.std(data_vector),
                         100 * np.percentile(data_vector, 1),
                         100 * np.percentile(data_vector, 5),
                         100 * np.percentile(data_vector, 10),
                         100 * np.percentile(data_vector, 25),
                         100 * np.percentile(data_vector, 50),
                         100 * np.percentile(data_vector, 75),
                         100 * np.percentile(data_vector, 90),
                         100 * np.percentile(data_vector, 95),
                         100 * np.percentile(data_vector, 99)]
    return dict(zip(statistics_names, statistics_vector))


def build_graphs(graph_data, time_data, flow_data, alpha, keys, filename, title, extension):
    fig, ax = plt.subplots()

    accumulated_time = [dt.datetime.fromtimestamp(ts) for ts in time_data]
    xfmt = md.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)

    ax.set_title(title)

    for key in keys:
        data_selected = 100 * np.asarray(graph_data[key][alpha])
        ax.plot(accumulated_time, data_selected, label=parse_name(key)['simple_name'])

    ax2 = ax.twinx()
    ax2.xaxis.set_major_formatter(xfmt)
    ax2.plot(accumulated_time, 8 * np.asarray(flow_data) / (1024 * 1024 * 1024), label='Flow', lw=7, alpha=0.4)
    ax2.set_ylabel('GbpS per second')

    fig.autofmt_xdate()

    ax.set_xlabel('Time HH-MM-SS')
    ax.set_ylabel(alpha + ' %')

    ax.legend()
    fig.savefig(filename, format=extension)

    plt.close(fig)


def build_barchart(names_info, statistics, keys, alpha, target_name, filename, extension):
    sizes = list(enumerate(set([names_info[key]['size_value'] for key in keys])))
    assert len(sizes) != 0
    sizes = sorted(sizes, key=lambda x: x[1])
    order, sizes = zip(*sizes)

    keys_lightweight = None

    for i in order:
        size = sizes[i]

        keys_simplified = sorted([names_info[key]['simple_name'] for key in
                                  keys if names_info[key]['size_value'] == size])

        assert len(keys_simplified) == len(set(keys_simplified))

        if keys_lightweight is None:
            keys_lightweight = keys_simplified
        else:
            if keys_simplified != keys_lightweight:
                print 'At comparison size is inconsistent with', keys_lightweight, keys_simplified
                assert False

    assert keys_lightweight is not None

    print '\tPrinting', filename
    fig, ax = plt.subplots()
    n_groups = len(sizes)
    n_types = len(keys_lightweight)
    scale = 0.8
    width = scale / n_types
    widths = np.linspace(-.5 + width/2 + (1 - scale) / 2, .5 - width/2 - (1 - scale) / 2, n_types)
    index = np.arange(n_groups)
    for i in range(len(keys_lightweight)):
        lwc = keys_lightweight[i]
        keys_for_lws = [(key, names_info[key]['size_value'])
                        for key in keys if names_info[key]['simple_name'] == lwc]
        keys_for_lws = zip(*sorted(keys_for_lws, key=lambda x: x[1]))[0]
        target = [statistics[key][alpha][target_name] for key in keys_for_lws]
        if target_name == 'mean':
            size_std = [statistics[key][alpha]['std'] for key in keys_for_lws]
        else:
            size_std = None
        ax.bar(index + widths[i], target, width, yerr=size_std, label=lwc, alpha=0.8)
    ax.set_ylabel(alpha + ' %')
    ax.set_title(alpha + ' ' + target_name)
    ax.set_xticks(index)
    ax.set_xticklabels([hurry_fsize(item * 1024 * 1024) for item in sizes])
    ax.set_xlabel('Size')
    ax.legend(loc=4)
    fig.savefig(filename, format=extension)
    plt.close(fig)


def build_percentiles(names_info, statistics, keys, alphas, output_folder, extension):
    for alpha in alphas:
        for stat_name in statistics_names:
            if stat_name == 'std':
                continue
            stat_name_fixed = stat_name.replace(' ', '').replace('%', '')
            filename = output_folder + alpha + '_' + stat_name_fixed + '.' + extension
            build_barchart(names_info, statistics, keys, alpha, stat_name, filename, extension)