import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as md
import datetime as dt


def build_graphs(x, y, label, prefix):
    plt.clf()
    fig, ax = plt.subplots()

    accumulated_time = [dt.datetime.fromtimestamp(ts) for ts in x]
    xfmt = md.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)

    ax.set_title(label)

    ax.plot(accumulated_time, y, label=label)

    fig.autofmt_xdate()

    ax.set_xlabel('Time HH-MM-SS')
    ax.set_ylabel(label)

    ax.legend(prop={'size': 3})

    fig.savefig(prefix + '/hist_' + label.replace(' ', '_') + '.png')
    plt.close(fig)


def build_hists(y, label, prefix, limits=None):
    plt.clf()
    fig, ax = plt.subplots()

    ax.set_title(label)

    ax.hist(y, label=label, bins=max(50, len(y) / 100), normed=True)

    ax.set_xlabel('Value')
    ax.set_ylabel(label)

    if limits is not None:
        xl, yl = limits
        ax.set_xlim(xl[0], xl[1])
        ax.set_ylim(yl[0], yl[1])

    ax.legend(prop={'size': 3})
    fig.savefig(prefix + '/hist_' + label + '.png')
    plt.close(fig)


def retrieve_data(folder):
    data = pd.read_csv(folder + '/data.csv')
    header = list(data)

    timestamp_name = header[0]
    feature_names = header[1:]

    data_dict = {}
    for name in feature_names:
        data_dict[name] = {}
        data_dict[name]['Mean'] = np.mean(data[name])
        data_dict[name]['STD'] = np.std(data[name])
        data_dict[name]['MoS'] = np.median(data[name])#data_dict[name]['Mean'] / data_dict[name]['STD']
        #build_graphs(data[timestamp_name], data[name], name, folder)

    return data_dict


parser = argparse.ArgumentParser(description='Test block')
parser.add_argument("experiment", type=str, help="Name of the experiment")

args = parser.parse_args()

names = args.experiment.split(' ')

data = {}

total_names = None

for name in names:

    data[name] = retrieve_data(name)
    if total_names is None:
        total_names = data[name].keys()
    else:
        assert set(total_names) == set(data[name].keys())

header = ['{:^25s}'.format('dataset')] + [' || {:^51s}'.format(name) for name in names]
header = ''.join(header)
header_local = ['{:^25s}'.format('feature')] + [' || {:^15s} | {:^15s} | {:^15s}'.format('Mean', 'STD', 'Median') for name in names]
header_local = ''.join(header_local)
filler_local_top = ['{:^25s}'.format(''.join(['='] * 25))] + \
                   ['=||={:^15s}==={:^15s}==={:^15s}'.format(''.join(['='] * 15),
                                                             ''.join(['='] * 15),
                                                             ''.join(['='] * 15)) for name in names]
filler_local_top = ''.join(filler_local_top)
filler_local_top_2 = ['{:^25s}'.format(''.join(['='] * 25))] + \
                     ['=||={:^15s}=|={:^15s}=|={:^15s}'.format(''.join(['='] * 15),
                                                               ''.join(['='] * 15),
                                                               ''.join(['='] * 15)) for name in names]
filler_local_top_2 = ''.join(filler_local_top_2)
filler_top = ''.join(['='] * len(header_local))
filler = ['{:^25s}'.format(''.join(['-'] * 25))] + ['-||-{:^15s}-|-{:^15s}-|-{:^15s}'.format(
    ''.join(['-'] * 15), ''.join(['-'] * 15), ''.join(['-'] * 15)) for name in names]
filler = ''.join(filler)
print 'O==' + filler_top + '==O'
print '|| ' + header + ' ||'
print '||=' + filler_local_top + '=||'
print '|| ' + header_local + ' ||'
print '||=' + filler_local_top_2 + '=||'

first = True
for feature_name in sorted(total_names):
    pstr = ['{:25s}'.format(feature_name)]
    for dataset_name in names:
        pstr.append(' || {:15.5f} | {:15.5f} | {:15.5f}'.format(
            data[dataset_name][feature_name]['Mean'],
            data[dataset_name][feature_name]['STD'],
            data[dataset_name][feature_name]['MoS']))
    if not first:
        print '||-' + filler + '-||'
    else:
        first = False
    print '|| ' + ''.join(pstr) + ' ||'
print 'O==' + filler_top + '==O'
