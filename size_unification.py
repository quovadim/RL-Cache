import pandas as pd
from os import listdir
from os.path import isfile, join
import argparse
from tqdm import tqdm


def iterate_dataset(filepath):
    filelist = sorted([filepath + f for f in listdir(filepath) if isfile(join(filepath, f)) and '.out' in f])
    for filename in sorted(filelist):
        print 'Using file', filename
        local_frame = pd.read_csv(filename, index_col=False, delimiter=' ', names=['timestamp', 'id', 'size', 'unused'])
        yield local_frame


parser = argparse.ArgumentParser(description='Tool to fix size issue into the source dataset')
parser.add_argument("data_path", type=str, help="Path to the source data")
parser.add_argument("output_path", type=str, help="Path to source csv")
parser.add_argument("output_file_storage", type=str, help="File that contains list of output files")

args = parser.parse_args()

start_series = None
counter = 0
id_size_frame = None
size_table = {}
for frame in iterate_dataset(args.data_path):
    frame.drop(columns=['timestamp', 'unused'], inplace=True)
    if start_series is None:
        frame = frame.groupby('id').max()
        frame.reset_index(inplace=True)
        start_series = frame
    else:
        start_series = pd.concat([start_series, frame])
    counter += 1


counter = 0
olf = open(args.output_file_storage, 'w')
lids = start_series['id'].values
lsize = start_series['size'].values
rep_dict = dict(zip(lids, lsize))
for frame in iterate_dataset(args.data_path):
    ids = frame['id'].values
    new_size = [0] * len(ids)
    for i in tqdm(range(len(new_size))):
        new_size[i] = rep_dict[ids[i]]
    frame['size'] = new_size
    frame.drop(columns='unused', inplace=True)
    frame.to_csv(args.output_path + str(counter) + '.csv', sep=' ', index=False, header=False)
    olf.write(args.output_path + str(counter) + '.csv\n')
    counter += 1
olf.close()
