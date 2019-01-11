import os
import argparse
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description='Algorithm tester')
parser.add_argument("dataset", type=str, help="dataset_suffix")
parser.add_argument("experiments", type=str, help="dataset_suffix")

parser.add_argument("-f", "--filename", type=str, default='0', help="Output filename")

parser.add_argument('-s', '--skip', type=int, default=0, help="Skip")
parser.add_argument('-m', '--smooth', type=int, default=1, help="Smooth factor")
parser.add_argument('-x', '--extension', type=str, default='pdf', help="Target extension")

parser.add_argument('-e', '--remove', action='store_true', help="Remove previous graphs")

parser.add_argument('-l', '--plots', action='store_true', help="Build sizes graphs")
parser.add_argument('-p', '--percentiles', action='store_true', help="Build size-aware percentiles")
parser.add_argument('-n', '--normed_plots', action='store_true', help="Build normalizes size-aware percentiles")

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
        assert time_data == time_data_new
        assert flow_data == flow_data_new
        if iterations_data is not None:
            if iterations_new is not None:
                assert iterations_data == iterations_new
        else:
            if iterations_new is not None:
                iterations_data = iterations_new

        items_to_remove = [item for item in set(reversal_mapping.keys() + reversal_mapping_new.keys())
                           if item not in reversal_mapping.keys() or item not in reversal_mapping_new.keys()]
        for item in items_to_remove:
            if item in reversal_mapping.keys():
                del reversal_mapping[item]
            if item in reversal_mapping_new.keys():
                del reversal_mapping_new[item]

        if reversal_mapping == {}:
            print 'Empty mapping'
            exit(0)

        new_keys = graph_data_new.keys()
        for key in new_keys:
            graph_data[key] = graph_data_new[key]
            statistics[key] = statistics_new[key]
            names_info[key] = names_info_new[key]

unique_periods = list(set(periods.values()))
assert len(unique_periods) == 1
period = unique_periods[0]

keys = graph_data.keys()

keys_rebuilt = [key for key in keys if
                names_info[key]['size'] > max_size or names_info[key]['size'] < min_size]
for key in keys_rebuilt:
    del graph_data[key]
    del statistics[key]
    del names_info[key]

keys = graph_data.keys()

if args.percentiles:
    print 'Building percentiles'
    percentiles_folder = target_data + 'percentiles/'
    if not os.path.exists(percentiles_folder):
        os.makedirs(percentiles_folder)
    build_percentiles(names_info, statistics, keys, reversal_mapping.keys(), percentiles_folder, extension)

graph_folder = target_data + 'graphs/'
if not os.path.exists(graph_folder):
    os.makedirs(graph_folder)

smoothed_graph_data = {}
for key in graph_data.keys():
    element_list = {}
    smoothed_graph_data[key] = {}
    for alpha in graph_data[key].keys():
        smoothed_graph_data[key][alpha] = smooth(graph_data[key][alpha], args.smooth,
                                                 iterations_data, flow_data, alpha, period)

smoothed_flow_data = smooth(None, args.smooth, None, flow_data, 0, period)
iterations_data = smooth(None, args.smooth, iterations_data, None, 0, period)
smoothed_entropy_data = smooth(entropy_data, args.smooth, None, None, 0, period)
smoothed_time_data = [int(item) for item in smooth(time_data, args.smooth, None, None, 0, period)]
period *= args.smooth

if args.plots:
    print 'Building graphs'
    sizes = list(set([names_info[key]['size'] for key in names_info.keys()]))
    sizes_mapping = {}
    for size in sizes:
        algorithms_to_build = [key for key in smoothed_graph_data.keys() if names_info[key]['size'] == size]
        for alpha in reversal_mapping.keys():
            filename = graph_folder + names_info[algorithms_to_build[0]]['text size'] + '_' + alpha + '.' + extension
            title = alpha + ' ' + names_info[algorithms_to_build[0]]['text size']
            print '\tBuilding', filename
            build_graphs(smoothed_graph_data, smoothed_time_data, smoothed_entropy_data,
                         alpha, algorithms_to_build, filename, title, extension)

if args.normed_plots:
    print 'Building relative graphs'
    min_key_perfix = u'AL-GDSF-'

    graph_folder = target_data + 'graphs_relative/'
    if not os.path.exists(graph_folder):
        os.makedirs(graph_folder)

    relative_graph_data = {}

    normalization_keys = {}
    for key in smoothed_graph_data.keys():
        if min_key_perfix in key:
            normalization_keys[names_info[key]['size']] = key

    for key in smoothed_graph_data.keys():
        element_list = {}
        relative_graph_data[key] = {}
        for alpha in smoothed_graph_data[key].keys():
            normalization_key = normalization_keys[names_info[key]['size']]
            relative_graph_data[key][alpha] = \
                np.asarray(smoothed_graph_data[key][alpha]) - np.asarray(smoothed_graph_data[normalization_key][alpha])

    sizes = list(set([names_info[key]['size'] for key in smoothed_graph_data.keys()]))
    sizes_mapping = {}
    for size in sizes:
        algorithms_to_build = [key for key in names_info.keys() if names_info[key]['size'] == size]
        for alpha in reversal_mapping.keys():
            filename = graph_folder + names_info[algorithms_to_build[0]]['text size'] + '_' + alpha + '.' + extension
            title = alpha + ' ' + names_info[algorithms_to_build[0]]['text size']
            print '\tBuilding', filename
            build_graphs(relative_graph_data, smoothed_time_data, smoothed_flow_data,
                         alpha, algorithms_to_build, filename, title, extension)