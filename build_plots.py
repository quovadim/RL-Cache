import os
import argparse
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description='Algorithm tester')
parser.add_argument("experiments", type=str, help="dataset_suffix")
parser.add_argument("dataset", type=str, help="dataset_suffix")

parser.add_argument("-f", "--filename", type=str, default='0', help="Output filename")

parser.add_argument('-s', '--skip', type=int, default=0, help="Skip")
parser.add_argument('-m', '--smooth', type=int, default=1, help="Smooth factor")
parser.add_argument('-x', '--extension', type=str, default='pdf', help="Target extension")

parser.add_argument('-e', '--remove', action='store_true', help="Remove previous graphs")

parser.add_argument('-l', '--plots', action='store_true', help="Build sizes graphs")
parser.add_argument('-p', '--percentiles', action='store_true', help="Build size-aware percentiles")

parser.add_argument('-b', '--background', type=str, default=None, help='Background plot')

args = parser.parse_args()

from graphs_auxiliary import smooth, build_graphs, build_percentiles, load_dataset, get_number_of_steps
from configuration_info.config_sanity import check_test_config
from configuration_info.filestructure import *

extension = args.extension

experiments = sorted(args.experiments.split(' '))

target_data = get_graphs_name('--'.join(experiments), args.dataset) + '/'

if args.remove:
    os.system('rm -rf ' + target_data)
    print target_data, 'removed'

if not os.path.exists(target_data):
    os.makedirs(target_data)

data = None

common_length = None

print 'Loading data...'

configs_loaded = []
for experiment in experiments:
    configuration = check_test_config(experiment, args.dataset, verbose=False)
    if configuration is None:
        exit(-1)
    configs_loaded.append(configuration)

    folder = configuration["output folder"] + '/'

    steps = get_number_of_steps(folder, args.filename, args.skip)
    if common_length is None or common_length > steps:
        common_length = steps

print 'Maximum number of files to use is', common_length

periods = {}

for i in range(len(configs_loaded)):
    configuration = configs_loaded[i]
    folder = configuration["output folder"] + '/'

    label = 'EXP' + experiments[i]
    periods[label] = configuration['period']

    keys_to_ignore = configuration['classical']

    data_new = load_dataset(folder, args.filename, args.skip, keys_to_ignore, common_length, uid=label)

    if data is None:
        data = data_new
    else:
        keys_to_equalize = ['time', 'flow', 'iterations', 'entropy']
        for key in keys_to_equalize:
            if key not in data.keys() and key in data_new.keys():
                data[key] = data_new[key]
        data['alphas'] = list(set(data['alphas']).intersection(set(data_new['alphas'])))
        for key in data_new['performance'].keys():
            data['performance'][key] = {}
            for alpha in data['alphas']:
                data['performance'][key][alpha] = data_new['performance'][key][alpha]
        for key in data['performance'].keys():
            for alpha in data['performance'][key].keys():
                if alpha not in data['alphas']:
                    del data['performance'][key][alpha]

unique_periods = list(set(periods.values()))
assert len(unique_periods) == 1
period = unique_periods[0]

if args.percentiles:
    print 'Building percentiles'
    percentiles_folder = target_data + 'percentiles/'
    if not os.path.exists(percentiles_folder):
        os.makedirs(percentiles_folder)
    build_percentiles(data, data['performance'].keys(), data['mapping'].keys(), percentiles_folder, extension)

graph_folder = target_data + 'graphs/'
if not os.path.exists(graph_folder):
    os.makedirs(graph_folder)

if args.smooth != 1:
    smoothed_data = smooth(data, args.smooth, period)
else:
    smoothed_data = data

if args.plots:
    print 'Building graphs'
    sizes = list(set([data['info'][key]['size'] for key in data['info'].keys()]))
    sizes_mapping = {}
    for size in sizes:
        algorithms_to_build = [key for key in data['info'].keys() if data['info'][key]['size'] == size]
        for alpha in data['mapping'].keys():
            filename = graph_folder + data['info'][algorithms_to_build[0]]['text size'] + '_' + alpha + '.' + extension
            title = alpha + ' ' + data['info'][algorithms_to_build[0]]['text size']
            print '\tBuilding', filename

            background_key = args.background
            if args.background is not None and args.background not in smoothed_data.keys():
                background_key = None

            build_graphs(smoothed_data, alpha, algorithms_to_build, filename, title, extension, background_key)
