import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Simple data generator')
parser.add_argument("data", type=str, help="Name of the dataset")
parser.add_argument('-n', '--names', type=int, default=None, help="Number of unique names")
parser.add_argument('-s', '--size', type=int, default=None, help="Average object size")
parser.add_argument('-e', '--seed', type=int, default=None, help="Random generator seed")
parser.add_argument('-r', '--requests', type=int, default=50000000, help="Number of requests to generate")
parser.add_argument('-f', '--rpf', type=int, default=1000000, help="Number of requests per file")

args = parser.parse_args()

import numpy as np
np.random.seed(args.seed)

requests = args.requests

if args.names is None:
    number_of_names = np.random.randint(requests / 1000, 2 * requests / 1000)
else:
    number_of_names = args.names


if args.size is None:
    size = 20 * 1024
else:
    size = args.size

out_path = os.path.join('data/', args.data)
if not os.path.exists(out_path):
    os.makedirs(out_path)

print 'Generating sizes'
names_size_mapping = np.random.randint(size - size / 2, size + size / 2, size=number_of_names)

file_counter = 0

for i in range(0, requests, args.rpf):
    reqs_to_generate = min(requests - i, args.rpf)
    print 'Generating', reqs_to_generate, 'requests. Done', i, 'out of', requests

    ids = np.random.randint(0, number_of_names, size=reqs_to_generate)
    timestamps = np.random.exponential(1, size=reqs_to_generate).astype(np.int)

    with open(os.path.join(out_path, str(file_counter) + '.csv'), 'w') as f:
        for j in tqdm(range(reqs_to_generate)):
            wstr = str(timestamps[j]) + ' ' + str(ids[j]) + ' ' + str(names_size_mapping[ids[j]]) + '\n'
            f.write(wstr)
    file_counter += 1