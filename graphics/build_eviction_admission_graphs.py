import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
import argparse
from graphs_auxiliary import *

parser = argparse.ArgumentParser(description='Build admission graphs')
parser.add_argument("network", type=str, help="Network name suffix")
parser.add_argument("label", type=str, help="Legend label")
parser.add_argument('-l', '--length', type=int, default=None, help='Maximal length to read')
parser.add_argument('-r', '--random', action='store_true', help='Use random mode')

args = parser.parse_args()

np.random.seed(27)

pdrop = 0
ldim = 11
models = [args.network]
model_names = [args.label]
model_names_mapping = dict(zip(models, model_names))
print 'Loading eviction predictions'
predictions_eviction = {}

lmax = args.length
if lmax is not None:
    lmax *= 1000
for model in models:
    print 'Loading', model_names_mapping[model]
    predictions_data = load_file('auxiliary/eviction_predictions_' + model, lmax=lmax)
    if lmax is None or len(predictions_data) < lmax:
        lmax = len(predictions_data)
    predictions_eviction[model] = predictions_data

predictions_clipped = {}
for key in predictions_eviction.keys():
    np.random.seed(27)
    if args.random:
        predictions_clipped[key] = sample_values(predictions_eviction[key][pdrop:lmax], ldim)
    else:
        predictions_clipped[key] = np.argmax(predictions_eviction[key][pdrop:lmax], axis=1)
predictions_eviction = predictions_clipped

print 'Loading admission predictions'
predictions_admission = {}
for model in models:
    print 'Loading', model_names_mapping[model]
    predictions_data = load_file('auxiliary/admission_predictions_' + model, lmax=lmax)
    if lmax is None or len(predictions_data) < lmax:
        lmax = len(predictions_data)
    predictions_admission[model] = predictions_data

predictions_clipped = {}
for key in predictions_admission.keys():
    np.random.seed(27)
    if args.random:
        predictions_clipped[key] = sample_values(predictions_admission[key][pdrop:lmax], 2)
    else:
        predictions_clipped[key] = np.argmax(predictions_admission[key][pdrop:lmax], axis=1)
predictions_admission = predictions_clipped

print 'Loading features'
features_data = load_file('auxiliary/features', lmax)
print 'Loading SH'
sh_data = load_file('auxiliary/sh_predictions', lmax)

GDSF_data = sh_data[:, 0]
sh_data = sh_data[:, 2]


features_data = features_data[pdrop:]
sh_data = sh_data[pdrop:]

sh_data = sh_data.reshape(-1)

print 'Data length', lmax

steps = 3

perc_values = [0] + [(i * 100) / steps for i in range(1, steps)] + [100]

percentiles_gdsf = get_percentiles(GDSF_data, steps)

percentiles_ml = {}
for key in predictions_eviction.keys():
    percentiles_ml[key] = get_percentiles(predictions_eviction[key], steps)

for item_to_discover in header:
    print 'Discovering', item_to_discover

    discover_index = header.index(item_to_discover)

    for i in range(0, steps):

        plt.clf()
        plt.title(label[discover_index] + '{:3d} {:3d}'.format(perc_values[i], perc_values[i+1]))

        current_feature = deepcopy(features_data[:, discover_index])
        current_feature = convertors[discover_index](current_feature)

        local_sh_data = deepcopy(sh_data)
        local_gdsf_data = deepcopy(GDSF_data)
        local_predictions_eviction = deepcopy(predictions_eviction)
        local_predictions_admission = deepcopy(predictions_admission)

        if drop_ohw_local[discover_index]:
            predictions_clipped = {}
            for key in local_predictions_eviction.keys():
                predictions_clipped[key] = local_predictions_eviction[key][sh_data != 0]
            local_predictions_eviction = predictions_clipped
            predictions_clipped = {}
            for key in local_predictions_admission.keys():
                predictions_clipped[key] = local_predictions_admission[key][sh_data != 0]
            local_predictions_admission = predictions_clipped

            current_feature = current_feature[sh_data != 0]
            local_sh_data = local_sh_data[sh_data != 0]
            local_gdsf_data = local_gdsf_data[sh_data != 0]

        x_min, x_max = min(current_feature), max(current_feature)

        plot_cum(current_feature, 'Total {:6d}K'.format(len(current_feature) / 1000))

        gdsf_lsf = current_feature[local_sh_data != 0]
        lcf = current_feature[local_sh_data != 0]
        evicted_gdsf = lcf[filter_on_percetile(gdsf_lsf, percentiles_gdsf[i], percentiles_gdsf[i+1])]
        print 'GDSF Len: {:7.3f}'.format(len(evicted_gdsf) * 100. / len(current_feature))
        plot_cum(evicted_gdsf, 'SH+GDSF {:4d}-{:4d} {:6d}K'.format(perc_values[i],
                                                                   perc_values[i+1],
                                                                   len(evicted_gdsf)/1000))

        for key in local_predictions_eviction.keys():
            lfd = current_feature[local_predictions_admission[key] != 0]
            ldd = local_predictions_eviction[key][local_predictions_admission[key] != 0]
            evicted_ml = lfd[filter_on_percetile(ldd, percentiles_ml[key][i], percentiles_ml[key][i + 1])]
            print key, 'Len: {:7.3f}'.format(len(evicted_ml) * 100. / len(current_feature))
            plot_cum(evicted_ml, model_names_mapping[key] + ' {:3d}-{:3d} {:6d}K'.format(perc_values[i],
                                                                                         perc_values[i + 1],
                                                                                         len(evicted_ml) / 1000))

        plt.xlabel(subscripts[header.index(item_to_discover)])
        plt.ylabel('Cumulative fraction of requests')
        plt.xlim(x_min, x_max)
        plt.legend(loc='lower right')
        plt.savefig('graphs/eviction+admission/' + item_to_discover + '_' + str(perc_values[i]) + '_' + str(perc_values[i+1]) + '.png')
