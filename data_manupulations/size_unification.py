import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import argparse


def iterate_dataset(filepath):
    filelist = sorted([filepath + f for f in listdir(filepath) if isfile(join(filepath, f)) and '.out' in f])
    for filename in sorted(filelist):
        print 'Total {:d} using file'.format(len(filelist)), filename
        local_frame = pd.read_csv(filename, index_col=False, delimiter=' ', names=['timestamp', 'id', 'size', 'response'])
        yield local_frame


parser = argparse.ArgumentParser(description='Tool to fix size issue into the source dataset')
parser.add_argument("data_path", type=str, help="Path to the source data")
parser.add_argument("output_path", type=str, help="Path to output data")

args = parser.parse_args()

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

start_series = None
for frame in iterate_dataset(args.data_path):
    frame.drop(columns=['timestamp', 'response'], inplace=True)
    if start_series is None:
        frame = frame.groupby('id').max()
        frame.reset_index(inplace=True)
        start_series = frame
    else:
        start_series = pd.concat([start_series, frame])
        start_series = start_series.groupby('id').max()
        start_series.reset_index(inplace=True)

counter = 0
for frame in iterate_dataset(args.data_path):
    frame = frame.merge(start_series, left_on='id', right_on='id', how='left')
    frame['size'] = frame[["size_x", "size_y"]].max(axis=1)
    frame.drop(columns=['response', 'size_y', 'size_x'], inplace=True)
    frame.to_csv(args.output_path + str(counter) + '.csv', sep=' ', index=False, header=False)
    counter += 1
