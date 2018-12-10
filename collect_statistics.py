import argparse
from FeatureExtractor import collect_features
from os import listdir
from os.path import isfile, join

parser = argparse.ArgumentParser(description='Tool to collect feature statistics')
parser.add_argument("filename", type=str, help="Output filename")
parser.add_argument('-r', "--region", type=str, default='china', help="Data region")
parser.add_argument('-i', '--iterations', type=int, default=50, help="Number of iterations per epoch")

args = parser.parse_args()

filepath = 'data/' + args.region + '_featured/'
filenames = [(join(filepath, f), int(f.replace('.csv', ''))) for f in listdir(filepath)
             if isfile(join(filepath, f)) and '.csv' in f and 'lock' not in f]
filenames = sorted(filenames, key=lambda x: x[1])
filenames = [item[0] for item in filenames]

collect_features('auxiliary/' + args.filename, 1000 * args.iterations, filenames)
