import argparse
from os import listdir
from os.path import isfile, join

from configuration_info.filestructure import get_data_name
from feature.extractor import collect_features

parser = argparse.ArgumentParser(description='Tool to collect feature statistics')
parser.add_argument("output", type=str, help="Target filename")
parser.add_argument('-r', "--region", type=str, default='china', help="Data region")
parser.add_argument('-i', '--iterations', type=int, default=50, help="Number of iterations per epoch")

args = parser.parse_args()

filepath = get_data_name(args.region)
filenames = [(join(filepath, f), int(f.replace('.csv', ''))) for f in listdir(filepath)
             if isfile(join(filepath, f)) and '.csv' in f and 'lock' not in f]
filenames = sorted(filenames, key=lambda x: x[1])
filenames = [item[0] for item in filenames]

collect_features(args.output, 1000000 * args.iterations, filenames)
