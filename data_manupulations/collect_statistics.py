import argparse
from feature.extractor import collect_features
from configuration_info.config_sanity import check_statistics_config
from configuration_info.filestructure import get_data_name
from os import listdir
from os.path import isfile, join

parser = argparse.ArgumentParser(description='Tool to collect feature statistics')
parser.add_argument("config", type=str, help="Target configuration file")
parser.add_argument('-r', "--region", type=str, default='china', help="Data region")
parser.add_argument('-i', '--iterations', type=int, default=50, help="Number of iterations per epoch")

args = parser.parse_args()
config = check_statistics_config(args.config)
assert config is not None

filepath = get_data_name(args.region)
filenames = [(join(filepath, f), int(f.replace('.csv', ''))) for f in listdir(filepath)
             if isfile(join(filepath, f)) and '.csv' in f and 'lock' not in f]
filenames = sorted(filenames, key=lambda x: x[1])
filenames = [item[0] for item in filenames]

collect_features(config['statistics'], 1000000 * args.iterations, filenames, config['usable names'])
