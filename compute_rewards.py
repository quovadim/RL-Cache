import pandas as pd
from os import listdir
from os.path import isfile, join
import argparse
from collections import deque
from tqdm import tqdm
import sys


def iterate_dataset(filepath):
    filenames = [(join(filepath, f), int(f.replace('.csv', ''))) for f in listdir(filepath)
                 if isfile(join(filepath, f)) and '.csv' in f and 'lock' not in f]
    filenames = sorted(filenames, key=lambda x: x[1])
    filenames = [item[0] for item in filenames]
    for fname in reversed(filenames[:20]):
        names = ['timestamp', 'id', 'size', 'frequency', 'lasp_app', 'log_time',
                 'exp_recency', 'exp_log']
        types = [int, int, int, int, int, int, float, float]
        hdlr = open(fname, 'r').readlines()

        for line in reversed(hdlr):
            lines_converted = line.split(' ')
            lines_converted = [types[i](lines_converted[i]) for i in range(len(types))]
            yield True, dict(zip(names, lines_converted))

        yield False, fname.replace(filepath, '')


parser = argparse.ArgumentParser(description='Tool to fix size issue into the source dataset')
#parser.add_argument("data_path", type=str, help="Path to the source data")
#parser.add_argument("output_path", type=str, help="Path to source csv")

args = parser.parse_args()

lookup_period = 300000

logical_time = 0

rows_in_log = deque()

computed_ratings_and_time = {}

save_time = None

forget_lambda = 0.999997

history = []

for go, row in iterate_dataset('data/china_featured/'):
    if logical_time % 1000 == 0:
        sys.stdout.write('\rIteration ' + str(logical_time) + " " + str(len(computed_ratings_and_time.keys())))
        sys.stdout.flush()
    if not go:
        ofname = open('data/china_rewarded/' + row, 'w')
        print ''
        for lrow, reward in tqdm(reversed(history), total=len(history)):
            data = [lrow['timestamp'], lrow['id'], lrow['size'], lrow['frequency'], lrow['lasp_app'], lrow['log_time'],
                    lrow['exp_recency'], lrow['exp_log'], reward]
            data = [str(item) for item in data]
            ofname.write(' '.join(data) + '\n')
        ofname.close()
        history = []
        continue
    rid = row['id']
    try:
        v, lt = computed_ratings_and_time[rid]
        v = 1 + v * (forget_lambda ** (logical_time - lt))
        computed_ratings_and_time[rid] = (v, logical_time)
    except:
        computed_ratings_and_time[rid] = (1, logical_time)
        v = 1
    if logical_time % lookup_period == 0:
        remove_keys = []
        for key in computed_ratings_and_time.keys():
            v, lt = computed_ratings_and_time[key]
            if logical_time - lt > 2 * lookup_period:
                remove_keys.append(key)
        for key in remove_keys:
            del computed_ratings_and_time[key]
    history.append((row, v))
    logical_time += 1

