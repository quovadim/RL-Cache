import os
import argparse
import numpy as np
from fractions import gcd
from math import floor
from copy import deepcopy

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description='Plots builder')
parser.add_argument("experiments", type=str, help="Experiment folder")
parser.add_argument("dataset", type=str, help="Dataset folder")

parser.add_argument("-f", "--filename", type=str, default='0', help="Output filename")

parser.add_argument('-s', '--skip', type=int, default=0, help="Skip")
parser.add_argument('-m', '--smooth', type=int, default=1, help="Smooth factor")
parser.add_argument('-x', '--extension', type=str, default='pdf', help="Target extension")

parser.add_argument('-e', '--remove', action='store_true', help="Remove previous graphs")

parser.add_argument('-l', '--plots', action='store_true', help="Build sizes graphs")
parser.add_argument('-p', '--percentiles', action='store_true', help="Build size-aware percentiles")
parser.add_argument('-n', '--normalize', action='store_true', help="Substrace worst algorithm from others")
parser.add_argument('-c', '--correlation', action='store_true', help="Print correlation with entropy")

parser.add_argument('-b', '--background', type=str, default=None, help='Background plot')

args = parser.parse_args()

from graphs_auxiliary import smooth, build_graphs, build_percentiles, load_dataset, get_number_of_steps
from configuration_info.config_sanity import check_test_config
from configuration_info.filestructure import *
from environment.environment_aux import to_ts

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
common_time = None

periods = {}
factors = {}

times_seq = {}

print 'Loading data...'

configs_loaded = []
for experiment in experiments:
    configuration = check_test_config(experiment, args.dataset, verbose=False)
    if configuration is None:
        exit(-1)
    configs_loaded.append(configuration)

    folder = configuration["output folder"] + '/'
    periods[experiment] = configuration['period']
    factors[experiment] = configuration['period']

    times_seq[experiment] = get_number_of_steps(folder, args.filename, args.skip) * periods[experiment]

smallest_time = None
for key in times_seq.keys():
    if smallest_time is None or smallest_time > times_seq[key]:
        smallest_time = times_seq[key]

steps_to_use = {}
for key in periods.keys():
    steps_to_use[key] = int(floor(1.0 * smallest_time / periods[key]))

unique_periods = list(set(periods.values()))

if len(unique_periods) != 1:
    lcm_value = reduce(lambda a, b: a * b / gcd(a, b), unique_periods)
    period = lcm_value
else:
    period = unique_periods[0]

for key in periods.keys():
    factors[key] = period / periods[key]

print 'Maximum number of files to use is', to_ts(smallest_time)
print 'Common interval', period

for i in range(len(configs_loaded)):
    configuration = configs_loaded[i]
    folder = configuration["output folder"] + '/'

    label = 'EXP' + experiments[i]

    data_new = load_dataset(folder, args.filename, args.skip, steps_to_use[experiments[i]], uid=label)
    if factors[experiments[i]] != 1:
        data_new = smooth(data_new, factors[experiments[i]], periods[experiments[i]])

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
            data['info'][key] = data_new['info'][key]
            data['statistics'][key] = data_new['statistics'][key]
            for alpha in data_new['performance'][key].keys():
                data['performance'][key][alpha] = data_new['performance'][key][alpha]

if args.correlation and 'entropy' in data.keys() and 'flow' in data.keys():
    print 'entropy', 'flow', np.corrcoef(data['flow'], data['entropy'])[0][1]
    keys_mapping = [(key, data['info'][key]['size']) for key in data['performance'].keys()]
    for key, size in sorted(keys_mapping, key=lambda x: x[1]):
        for alpha in data['performance'][key].keys():
            print 'entropy', key, alpha, np.corrcoef(data['performance'][key][alpha], data['entropy'])[0][1]


HR_data = smooth(deepcopy(data), len(data['time']), period)

performances = {}
for key in HR_data['performance'].keys():
    for alpha in HR_data['performance'][key].keys():
        if alpha not in performances.keys():
            performances[alpha] = {}
        key_size = HR_data['info'][key]['size']
        if key_size not in performances[alpha].keys():
            performances[alpha][key_size] = {}
        performances[alpha][key_size][key] = HR_data['performance'][key][alpha][0]

for alpha in performances.keys():
    for sv in performances[alpha].keys():
        for key in performances[alpha][sv].keys():
            print alpha, key, performances[alpha][sv][key]

if args.percentiles:
    print 'Building percentiles'
    percentiles_folder = target_data + 'percentiles/'
    if not os.path.exists(percentiles_folder):
        os.makedirs(percentiles_folder)
    build_percentiles(data, data['performance'].keys(), data['mapping'].keys(), percentiles_folder, extension)

if args.plots:
    graph_folder = target_data + 'graphs/'
    if not os.path.exists(graph_folder):
        os.makedirs(graph_folder)

if args.normalize:
    normed_graph_folder = target_data + 'normed_graphs/'
    if not os.path.exists(normed_graph_folder):
        os.makedirs(normed_graph_folder)

if args.smooth != 1:
    smoothed_data = smooth(data, args.smooth, period)
    if args.correlation and 'entropy' in smoothed_data.keys() and 'flow' in smoothed_data.keys():
        print 'Smooth', 'entropy', 'flow', np.corrcoef(smoothed_data['flow'], smoothed_data['entropy'])[0][1]
        keys_mapping = [(key, smoothed_data['info'][key]['size']) for key in smoothed_data['performance'].keys()]
        for key, size in sorted(keys_mapping, key=lambda x: x[1]):
            for alpha in smoothed_data['performance'][key].keys():
                print 'Smooth', 'entropy', key, alpha, np.corrcoef(smoothed_data['performance'][key][alpha],
                                                         smoothed_data['entropy'])[0][1]
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

min_keys = {}
for alpha in performances.keys():
    for sv in performances[alpha].keys():
        mkey = None
        mval = None
        for key in performances[alpha][sv].keys():
            if mval is None or mval > performances[alpha][sv][key]:
                mval = performances[alpha][sv][key]
                mkey = key
        for key in performances[alpha][sv].keys():
            if key not in min_keys.keys():
                min_keys[key] = {}
            min_keys[key][alpha] = mkey

if args.normalize:
    print 'Building normalized graphs'
    n_data = {}
    for key in smoothed_data['performance'].keys():
        n_data[key] = {}
        for alpha in smoothed_data['performance'][key].keys():
            mkey = min_keys[key][alpha]
            n_data[key][alpha] = np.asarray(smoothed_data['performance'][key][alpha]) - \
                                 np.asarray(smoothed_data['performance'][mkey][alpha])
    smoothed_data['performance'] = n_data
    sizes = list(set([data['info'][key]['size'] for key in data['info'].keys()]))
    sizes_mapping = {}
    for size in sizes:
        algorithms_to_build = [key for key in data['info'].keys() if data['info'][key]['size'] == size]
        for alpha in data['mapping'].keys():
            filename = normed_graph_folder + data['info'][algorithms_to_build[0]]['text size'] + '_' + alpha + '.' + \
                       extension
            title = alpha + ' ' + data['info'][algorithms_to_build[0]]['text size']
            print '\tBuilding', filename

            background_key = args.background
            if args.background is not None and args.background not in smoothed_data.keys():
                background_key = None

            build_graphs(smoothed_data, alpha, algorithms_to_build, filename, title, extension, background_key)