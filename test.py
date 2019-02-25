import argparse

parser = argparse.ArgumentParser(description='Test block')
parser.add_argument("experiment", type=str, help="Name of the experiment")
parser.add_argument("test", type=str, help="Name of the test")
parser.add_argument('-g', '--gpu', action='store_true', help="Use GPU for computations")
parser.add_argument('-l', '--load', action='store_true', help="Load from dump")
parser.add_argument('-r', '--request', action='store_true', help="Load from dump")
parser.add_argument('-m', '--memopt', type=int, default=-1, help="Memory optimization period")

args = parser.parse_args()

import os

from environment.environment import test
from configuration_info.config_sanity import check_test_config
from configuration_info.filestructure import get_test_dump_name

if not args.gpu:
    print '=============Ignoring GPU============='
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

configuration = check_test_config(args.experiment, args.test, verbose=False)
configuration['memopt'] = args.memopt
if configuration is None:
    exit(0)

if not os.path.exists(configuration["output folder"]):
    os.makedirs(configuration["output folder"])

test(configuration, configuration["output folder"] + '/0',
     get_test_dump_name(args.experiment, args.test), args.load, args.request)
