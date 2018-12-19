import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import argparse

from environment.environment_aux import to_ts


def iterate_dataset(filelist):
    for filename in sorted(filelist):
        local_frame = pd.read_csv(filename, index_col=False, delimiter=' ', names=['timestamp', 'id', 'size',
                                                                                   'response'])
        yield filename, local_frame


parser = argparse.ArgumentParser(description='Tool to fix size issue into the source dataset')
parser.add_argument("data_path", type=str, help="Path to the source data")
parser.add_argument("output_path", type=str, help="Path to output data")
parser.add_argument("period", type=int, help="Period in days")
parser.add_argument("step", type=int, help="Step in days")
parser.add_argument("-m", "--mapping", type=str, default=None, help="Loading path to size mapping")

args = parser.parse_args()

filelist = sorted([args.data_path + f for f in listdir(args.data_path) if isfile(join(args.data_path, f))])

min_time = None
max_time = None

delete_mapping = False
mapping_name = args.output_path + '_' + 'mapping'
if args.mapping is None:
    python_args = [args.data_path,
                   '-m',
                   '-s=' + mapping_name]

    command = 'python data_manupulations/size_unification.py ' + ' '.join(python_args)
    delete_mapping = True
else:
    mapping_name = args.mapping

period = args.period * 24 * 60 * 60
step = args.step * 24 * 60 * 60

last_counter_save = 0

indicies = []
counters = []

counter = 0
for filename, frame in iterate_dataset(filelist):
    start_moment = frame.ix[5000, 'timestamp']
    end_moment = frame.ix[frame.shape[0] - 1, 'timestamp']
    if min_time is None:
        min_time = start_moment
        max_time = end_moment
    else:
        max_time = max(end_moment, max_time)

    if end_moment - last_counter_save > step:
        counters.append((start_moment, end_moment, counter))
        last_counter_save = start_moment

    new_counters = []
    for cs, ce, index in counters:
        if end_moment - cs > period:
            indicies.append((index, counter))
        else:
            new_counters.append((cs, end_moment, index))
    counters = new_counters

    print filename, '\033[1m' + to_ts(end_moment - start_moment), '\033[0m', \
        'Total \033[1m' + to_ts(max_time - min_time), '\033[0m', 'Periods gathered\033[1m', len(indicies), '\033[0m'
    counter += 1

new_counters = []
for cs, ce, index in counters:
    if ce - cs > period / 2:
        indicies.append((index, counter))
        break

data_index = 0
for s, e in indicies:
    python_args = [args.data_path,
                   '-o=' + args.output_path + str(data_index) + '_mapping/',
                   '-b=' + str(s),
                   '-e=' + str(e),
                   '-l=' + mapping_name]

    if not os.path.exists(args.output_path + str(data_index) + '_mapping/'):
        os.makedirs(args.output_path + str(data_index) + '_mapping/')

    command = 'python data_manupulations/size_unification.py ' + ' '.join(python_args)
    print 'Gathering sizes with ', command
    os.system(command)

    cpp_args = [
        args.output_path + str(data_index) + '_mapping/',
        args.output_path + str(data_index) + '/',
        str(e - s)
    ]
    if not os.path.exists(args.output_path + str(data_index) + '/'):
        os.makedirs(args.output_path + str(data_index) + '/')
    cpp_command = './feature_collector/collector ' + ' '.join(cpp_args)
    print 'Collecting features with', cpp_command
    os.system(cpp_command)

    os.system('rm -rf ' + args.output_path + str(data_index) + '_mapping/')
    data_index += 1

if delete_mapping:
    os.system('rm ' + mapping_name)
