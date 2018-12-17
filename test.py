import argparse
import os

from environment.environment import test
from configuration_info.config_sanity import check_test_config

parser = argparse.ArgumentParser(description='Algorithm trainer')
parser.add_argument("experiment", type=str, help="Name of the experiment")
parser.add_argument("test", type=str, help="Name of the test")
parser.add_argument('-g', '--gpu', action='store_true', help="Use GPU for computations")

args = parser.parse_args()

if not args.gpu:
    print '=============Ignoring GPU============='
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

configuration = check_test_config(args.experiment, args.test, verbose=False)
if configuration is None:
    exit(0)

if not os.path.exists(configuration["output folder"]):
    os.makedirs(configuration["output folder"])

test(configuration, configuration["output folder"] + '/0')
