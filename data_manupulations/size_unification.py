import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import argparse
import sys
from tqdm import tqdm
import numpy as np


def merge_dicts(dict1, dict2):
    merged_dict = {}
    for key in tqdm(set(dict1).union(dict2)):
        size_dict_1 = dict1.get(key, {})
        size_dict_2 = dict2.get(key, {})
        size_merged_dict = dict((k, size_dict_2.get(k, 0) + size_dict_1.get(k, 0))
                                for k in set(size_dict_1).union(size_dict_2))
        merged_dict[key] = size_merged_dict
    return merged_dict


def convert_to_median(source_dict):
    result = {}
    for key in tqdm(source_dict.keys()):
        size_distribution_dict = source_dict[key]
        median_array = sum([[key] * size_distribution_dict[key] for key in size_distribution_dict.keys()], [])
        result[key] = np.median(median_array)
    return result


def iterate_dataset(filelist):
    for filename in sorted(filelist):
        print filename
        local_frame = pd.read_csv(filename, index_col=False, delimiter=' ', names=['timestamp', 'id', 'size'])
        yield filename, local_frame[['timestamp', 'id', 'size']]


parser = argparse.ArgumentParser(description='Tool to fix size issue into the source dataset')
parser.add_argument("data_path", type=str, help="Path to the source data")
parser.add_argument("-o", "--output_path", type=str, default=None, help="Path to output data")
parser.add_argument("-b", "--begin", type=int, default=0, help="Number of files to skip from the beginning")
parser.add_argument("-e", "--end", type=int, default=None, help="Number of files to skip from the end")
parser.add_argument("-l", "--load", type=str, default=None,
                    help="Loading path to size mapping, will be created if none")
parser.add_argument("-s", "--save", type=str, default=None, help="Path to save size mapping")
parser.add_argument('-m', '--mapping_only', action='store_true', help="Collect only size mapping")

args = parser.parse_args()

filelist = sorted([args.data_path + f for f in listdir(args.data_path) if isfile(join(args.data_path, f))])

begin = args.begin
end = len(filelist) if args.end is None else args.end

filelist = filelist[begin:end]

size_dict = {}
if args.load is None:
    counter = 0
    total_lines = 0
    for filename, frame in iterate_dataset(filelist):
        frame.drop(columns=['timestamp'], inplace=True)
        total_lines += len(frame)

        print 'Doing {:d}/{:d}, lines: {:d}M, file {:s}'.format(1 + counter, len(filelist), int(total_lines/1e6),
                                                                filename)

        print 'Grouping...'
        frame = frame.groupby('id')

        print 'Collecting...'
        local_size_dict = {}
        for key in tqdm(frame.groups.keys()):
            values, counts = np.unique(frame.groups[key], return_counts=True)
            values, counts = list(values), list(counts)
            local_size_dict[key] = dict(zip(values, counts))

        print 'Merging...'
        size_dict = merge_dicts(size_dict, local_size_dict)

        if counter >= 5:
            break

        counter += 1

    print 'Calculating sequence'
    size_dict_aggregated = convert_to_median(size_dict)

    print 'Creating data frame'
    size_mapping = pd.DataFrame(size_dict.items(), columns=['id', 'size'])

    print 'Collected'
else:
    size_mapping = pd.read_csv(args.load)
    print 'Mapping loaded from', args.load

if args.save is not None:
    if args.load is not None:
        print 'You are saving loaded size mapping, it might be not the best idea'
    size_mapping.to_csv(args.save)
    print 'Mapping saved to', args.save

assert size_mapping is not None

if args.mapping_only:
    if args.load is not None or args.save is None:
        print 'You are doing something wrong'
        if args.load is not None:
            print "\tYou've loaded mapping and not fixing sizes"
        if args.save is None:
            print "\tYou are not saving mapping and not doing anything else"
    exit(0)

assert args.output_path is not None

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

counter = 0
total_lines = 0
for filename, frame in iterate_dataset(filelist):
    frame = frame.merge(size_mapping, left_on='id', right_on='id', how='left')
    frame['size'] = frame[["size_x", "size_y"]].max(axis=1)
    frame = frame[['timestamp', 'id', 'size']]
    total_lines += len(frame)
    sys.stdout.write('\rMerge {:d}/{:d} lines: {:d}M file {:s}'.format(
        1 + counter, len(filelist), int(total_lines / 1e6), args.output_path + str(counter) + '.csv'))
    sys.stdout.flush()
    frame.to_csv(args.output_path + str(counter) + '.csv', sep=' ', index=False, header=False)
    counter += 1

print ''
print 'Merged'
