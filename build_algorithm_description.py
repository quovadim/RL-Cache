import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description='Algorithm tester')
parser.add_argument("experiment", type=str, help="Experiment id")
parser.add_argument("region", type=str, help="Region id")

parser.add_argument("-f", "--filename", type=str, default='0', help="Output filename")

parser.add_argument('-s', '--skip', type=int, default=0, help="Skip")
parser.add_argument('-m', '--smooth', type=int, default=1, help="Smooth factor")
parser.add_argument('-x', '--extension', type=str, default='pdf', help="Target extension")

parser.add_argument('-e', '--remove', action='store_true', help="Remove previous graphs")

parser.add_argument('-l', '--plots', action='store_true', help="Build sizes graphs")
parser.add_argument('-p', '--percentiles', action='store_true', help="Build size-aware percentiles")

args = parser.parse_args()

from graphs_auxiliary import smooth, build_graphs, build_percentiles, load_dataset
from configuration_info.config_sanity import check_test_config
from configuration_info.filestructure import *


configuration = check_test_config(args.experiment, args.region, verbose=False)
if configuration is None:
    exit(-1)

extension = args.extension

output_folder = get_graphs_name(args.experiment, args.region) + '/'

if args.remove:
    os.system('rm -rf ' + output_folder)
    print output_folder, 'removed'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

folder = get_tests_name(args.experiment, args.region) + '/'

keys_to_ignore = configuration['classical']

print 'Loading data...'

graph_data, time_data, flow_data, iterations_data, reversal_mapping, names_info, statistics = load_dataset(
    folder, args.filename, args.skip, keys_to_ignore)

if args.percentiles:
    percentile_keys = [key for key in graph_data.keys() if key not in keys_to_ignore]
    print 'Building percentiles'
    percentiles_folder = output_folder + 'percentiles/'
    if not os.path.exists(percentiles_folder):
        os.makedirs(percentiles_folder)
    build_percentiles(names_info, statistics, percentile_keys, reversal_mapping.keys(), percentiles_folder, extension)

if not args.plots:
    exit(0)

graph_folder = output_folder + 'graphs/'
if not os.path.exists(graph_folder):
    os.makedirs(graph_folder)

smoothed_graph_data = {}
for key in graph_data.keys():
    element_list = {}
    smoothed_graph_data[key] = {}
    for alpha in graph_data[key].keys():
        smoothed_graph_data[key][alpha] = smooth(graph_data[key][alpha], args.smooth, iterations_data, flow_data,
                                                 reversal_mapping[alpha], configuration['period'])

smoothed_flow_data = smooth(None, args.smooth, None, flow_data, 0, configuration['period'])
iterations_data = smooth(None, args.smooth, iterations_data, None, 0, configuration['period'])
smoothed_time_data = [int(item) for item in smooth(time_data, args.smooth, None, None, 0, configuration['period'])]
configuration['period'] *= args.smooth

print 'Building graphs'
sizes = list(set([names_info[key]['size'] for key in names_info.keys()]))
sizes_mapping = {}
for size in sizes:
    algorithms_to_build = [key for key in names_info.keys() if names_info[key]['size'] == size]
    for alpha in reversal_mapping.keys():
        filename = graph_folder + names_info[algorithms_to_build[0]]['text size'] + '_' + alpha + '.' + extension
        title = alpha + ' ' + names_info[algorithms_to_build[0]]['text size']
        print '\tBuilding', filename
        build_graphs(smoothed_graph_data, smoothed_time_data, smoothed_flow_data,
                     alpha, algorithms_to_build, filename, title, extension)

