import argparse
from collections import deque
from os import listdir
from os.path import isfile, join
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Eviction feature extraction')
parser.add_argument('-r', "--region", type=str, default='china', help="Data region")
parser.add_argument('-l', '--length', type=int, default=300, help="Trace length")

args = parser.parse_args()

np.random.seed(27)

filepath = 'data/' + args.region + '_featured/'
filenames = [(join(filepath, f), int(f.replace('.csv', ''))) for f in listdir(filepath)
             if isfile(join(filepath, f)) and '.csv' in f and 'lock' not in f]
filenames = sorted(filenames, key=lambda x: x[1])
filenames = [item[0] for item in filenames]

from GameEnvironment import GameEnvironment
from FeatureExtractor import PacketFeaturer

iterations = 1000 * args.length

future_period = 300000
rows = deque(maxlen=future_period)
occurencies = {}

occurencies_data = []
gamma = 0.999
steps = 0

features = []

ofile = open('trace_china_2.tr', 'w')
counter = 0
for row in tqdm(GameEnvironment.iterate_dataset_over_all(filenames, 0), total=60000000):
    ofile.write(str(row['timestamp']) + ' ' + str(row['id']) + ' ' + str(row['size']) + '\n')
    counter += 1
    if counter == 60000000:
        break
ofile.close()