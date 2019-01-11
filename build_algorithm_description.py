import os
import argparse
import numpy as np

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

parser.add_argument('-b', '--background', type=str, default=None, help='Background plot')

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

data = load_dataset(folder, args.filename, args.skip, keys_to_ignore)

if 'entropy' in data.keys() and 'flow' in data.keys():
    print 'entropy', 'flow', np.corrcoef(data['flow'], data['entropy'])[0][1]
    keys_mapping = [(key, data['info'][key]['size']) for key in data['performance'].keys()]
    for key, size in sorted(keys_mapping, key=lambda x: x[1]):
        for alpha in data['performance'][key].keys():
            print 'entropy', key, alpha, np.corrcoef(data['performance'][key][alpha], data['entropy'])[0][1]

if args.percentiles:
    percentile_keys = [key for key in data['performance'].keys() if key not in keys_to_ignore]
    print 'Building percentiles'
    percentiles_folder = output_folder + 'percentiles/'
    if not os.path.exists(percentiles_folder):
        os.makedirs(percentiles_folder)
    build_percentiles(data, percentile_keys, data['mapping'].keys(), percentiles_folder, extension)

if not args.plots:
    exit(0)

graph_folder = output_folder + 'graphs/'
if not os.path.exists(graph_folder):
    os.makedirs(graph_folder)

if args.smooth != 1:
    smoothed_data = smooth(data, args.smooth, configuration['period'])
else:
    smoothed_data = data

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

