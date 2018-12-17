import argparse
import os

from GameEnvironment import test
from config_sanity import check_test_config

parser = argparse.ArgumentParser(description='Algorithm trainer')
parser.add_argument("config", type=str, help="Configuration file for training")
parser.add_argument('-g', '--gpu', action='store_true', help="Use GPU for computations")

args = parser.parse_args()

if not args.gpu:
    print '=============Ignoring GPU============='
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

configuration = check_test_config(args.config, verbose=False)
if configuration is None:
    exit(0)

if not os.path.exists(configuration["output_folder"]):
    os.makedirs(configuration["output_folder"])

test(configuration, configuration["output_folder"] + '/0')
