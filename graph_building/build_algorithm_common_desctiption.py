import os
import argparse
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from graphs_auxiliary import smooth, build_graphs, build_percentiles, load_dataset, get_number_of_steps
from configuration_info.config_sanity import check_test_config


parser = argparse.ArgumentParser(description='Algorithm tester')
parser.add_argument("dataset", type=str, help="dataset_suffix")
parser.add_argument("output_folder", type=str, help="output folder")

parser.add_argument("-f", "--filename", type=str, default='0', help="Output filename")

parser.add_argument('-s', '--skip', type=int, default=0, help="Skip")
parser.add_argument('-m', '--smooth', type=int, default=1, help="Smooth factor")
parser.add_argument('-x', '--extension', type=str, default='pdf', help="Target extension")

parser.add_argument('-e', '--remove', action='store_true', help="Remove previous graphs")

parser.add_argument('-l', '--plots', action='store_true', help="Build sizes graphs")
parser.add_argument('-p', '--percentiles', action='store_true', help="Build size-aware percentiles")
parser.add_argument('-n', '--normed_plots', action='store_true', help="Build normalizes size-aware percentiles")

args = parser.parse_args()

extension = args.extension

if args.remove:
    os.system('rm -rf ' + args.output_folder)
    print args.output_folder, 'removed'

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)
#'LRU', 'LFU',
algorithms = ['GDSF', 'Q']

configs = ['configs/M' + item + '_test_' + args.dataset + '.json' for item in algorithms]

graph_data = None
time_data = []
flow_data = []
alphas = []
names_info = {}
statistics = {}

mlength = None

max_size = 16 * 1024
min_size = 0

print 'Loading data...'

for config in configs:
    configuration = check_test_config(config, verbose=False, load_only=True)
    if configuration is None:
        exit(-1)

    folder = configuration["output_folder"] + '/'

    steps = get_number_of_steps(folder, args.filename, args.skip)
    if mlength is None or mlength > steps:
        mlength = steps

print 'Maximum number of files to use is', mlength

for config in configs:
    configuration = check_test_config(config, verbose=False, load_only=True)
    if configuration is None:
        exit(-1)

    folder = configuration["output_folder"] + '/'

    keys_to_ignore = []
    if "generic checker" in configuration.keys():
        keys_to_ignore += configuration["generic checker"]

    graph_data_new, time_data_new, flow_data_new, alphas_new, names_info_new, statistics_new = load_dataset(
        folder, args.filename, args.skip, keys_to_ignore, mlength)

    if graph_data is None:
        graph_data = graph_data_new
        time_data = time_data_new
        flow_data = flow_data_new
        alphas = alphas_new
        names_info = names_info_new
        statistics = statistics_new
    else:
        assert time_data == time_data_new
        assert flow_data == flow_data_new
        alphas = [item for item in alphas if item in alphas_new]
        new_keys = graph_data_new.keys()
        for key in new_keys:
            graph_data[key] = graph_data_new[key]
            statistics[key] = statistics_new[key]
            names_info[key] = names_info_new[key]

keys = graph_data.keys()

keys_rebuilt = [key for key in keys if
                names_info[key]['size_value'] > max_size or names_info[key]['size_value'] < min_size]
for key in keys_rebuilt:
    del graph_data[key]
    del statistics[key]
    del names_info[key]

keys = graph_data.keys()

if args.percentiles:
    print 'Building percentiles'
    percentiles_folder = args.output_folder + 'percentiles/'
    if not os.path.exists(percentiles_folder):
        os.makedirs(percentiles_folder)
    build_percentiles(names_info, statistics, keys, alphas, percentiles_folder, extension)

graph_folder = args.output_folder + 'graphs/'
if not os.path.exists(graph_folder):
    os.makedirs(graph_folder)

smoothed_time_data = [int(item) for item in smooth(time_data, args.smooth)]
smoothed_flow_data = smooth(flow_data, args.smooth)

smoothed_graph_data = {}
for key in graph_data.keys():
    element_list = {}
    smoothed_graph_data[key] = {}
    for alpha in graph_data[key].keys():
        smoothed_graph_data[key][alpha] = smooth(graph_data[key][alpha], args.smooth)

if args.plots:
    print 'Building graphs'
    sizes = list(set([names_info[key]['size_value'] for key in names_info.keys()]))
    sizes_mapping = {}
    for size in sizes:
        algorithms_to_build = [key for key in names_info.keys() if names_info[key]['size_value'] == size]
        for alpha in alphas:
            filename = graph_folder + names_info[algorithms_to_build[0]]['size'] + '_' + alpha + '.' + extension
            title = alpha + ' ' + names_info[algorithms_to_build[0]]['size']
            print '\tBuilding', filename
            build_graphs(smoothed_graph_data, smoothed_time_data, smoothed_flow_data,
                         alpha, algorithms_to_build, filename, title, extension)

if args.normed_plots:
    print 'Building relative graphs'
    min_key_perfix = u'AL-GDSF-A-'

    graph_folder = args.output_folder + 'graphs_relative/'
    if not os.path.exists(graph_folder):
        os.makedirs(graph_folder)

    relative_graph_data = {}
    for key in smoothed_graph_data.keys():
        element_list = {}
        relative_graph_data[key] = {}
        for alpha in smoothed_graph_data[key].keys():
            normalization_key = min_key_perfix + str(names_info[key]['size_value'])
            relative_graph_data[key][alpha] = \
                np.asarray(smoothed_graph_data[key][alpha]) - np.asarray(smoothed_graph_data[normalization_key][alpha])

    sizes = list(set([names_info[key]['size_value'] for key in names_info.keys()]))
    sizes_mapping = {}
    for size in sizes:
        algorithms_to_build = [key for key in names_info.keys() if names_info[key]['size_value'] == size]
        for alpha in alphas:
            filename = graph_folder + names_info[algorithms_to_build[0]]['size'] + '_' + alpha + '.' + extension
            title = alpha + ' ' + names_info[algorithms_to_build[0]]['size']
            print '\tBuilding', filename
            build_graphs(relative_graph_data, smoothed_time_data, smoothed_flow_data,
                         alpha, algorithms_to_build, filename, title, extension)