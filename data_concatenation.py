import argparse
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from GameEnvironment import GameEnvironment
import sys

parser = argparse.ArgumentParser(description='Data concatenation')
parser.add_argument("input", type=str, help="Path to the data")
parser.add_argument("output", type=str, help="Output file")
parser.add_argument('-l', '--length', type=int, default=400, help="Trace length")

args = parser.parse_args()

filepath = args.input
filenames = [(join(filepath, f), int(f.replace('.csv', ''))) for f in listdir(filepath) if isfile(join(filepath, f)) and '.csv' in f and 'lock' not in f]
filenames = sorted(filenames, key=lambda x: x[1])
filenames = [item[0] for item in filenames]

env = GameEnvironment(filepath, 0, skip=0)

rows = []

counter = 0

output_file = open(args.output, 'w')

iterations = args.length * 1000

for row in tqdm(GameEnvironment.iterate_dataset_over_all(filenames, 0), total=iterations):
    rows.append(row)

    if len(rows) == 1000:
        for r in rows:
            rstr = ' '.join([str(r['timestamp']), str(r['id']), str(r['size'])]) + '\n'
            output_file.write(rstr)
        rows = []

        counter += 1000

    if counter >= iterations:
        break
