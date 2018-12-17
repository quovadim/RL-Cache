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
models = [args.network]
model_names = [args.label]
model_names_mapping = dict(zip(models, model_names))
print 'Loading predictions'
predictions = {}
lmax = args.length
if lmax is not None:
    lmax *= 1000
for model in models:
    print 'Loading', model_names_mapping[model]
    predictions_data = load_file('auxiliary/admission_predictions_' + model, lmax=lmax)
    if lmax is None or len(predictions_data) < lmax:
        lmax = len(predictions_data)
    predictions[model] = predictions_data
predictions_clipped = {}
for key in predictions.keys():
    np.random.seed(27)
    if args.random:
        predictions_clipped[key] = sample_values(predictions[key][pdrop:lmax], 2)
    else:
        predictions_clipped[key] = np.argmax(predictions[key][pdrop:lmax], axis=1)
predictions = predictions_clipped
print 'Loading features'
features_data = load_file('auxiliary/features', lmax)
print 'Loading SH'
sh_data = load_file('auxiliary/sh_predictions', lmax)
sh_data = sh_data[:, 2]

features_data = features_data[pdrop:]
sh_data = sh_data[pdrop:]

sh_data = sh_data.reshape(-1)

print 'Data length', lmax

for item_to_discover in header:
    print 'Discovering', item_to_discover

    discover_index = header.index(item_to_discover)

    plt.clf()
    plt.title(label[discover_index])

    current_feature = deepcopy(features_data[:, discover_index])
    current_feature = convertors[discover_index](current_feature)

    local_sh_data = deepcopy(sh_data)
    local_predictions = deepcopy(predictions)

    if drop_ohw_local[discover_index]:
        predictions_clipped = {}
        for key in local_predictions.keys():
            predictions_clipped[key] = local_predictions[key][sh_data != 0]
        local_predictions = predictions_clipped

        current_feature = current_feature[sh_data != 0]
        local_sh_data = sh_data[sh_data != 0]

    admitted_by_secondhit = current_feature[local_sh_data != 0]
    plot_cum(admitted_by_secondhit, 'SH+GDSF {:6d}K'.format(len(admitted_by_secondhit)/1000))

    plot_cum(current_feature, 'Total {:6d}K'.format(len(current_feature)/1000))

    for key in local_predictions.keys():
        admitted_by_ml = current_feature[local_predictions[key] != 0]
        plot_cum(admitted_by_ml, model_names_mapping[key] + ' {:6d}K'.format(len(admitted_by_ml)/1000))

    plt.xlabel(subscripts[header.index(item_to_discover)])
    plt.ylabel('Cumulative fraction of requests')
    plt.legend(loc='lower right')
    plt.savefig('graphs/admission/' + item_to_discover + '.png')
