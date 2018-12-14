import argparse
import os
from os import listdir
from os.path import isfile, join
import json

from GameEnvironment import test
from config_sanity import check_test_config

parser = argparse.ArgumentParser(description='Algorithm trainer')
parser.add_argument("config", type=str, help="Configuration file for training")
parser.add_argument("generator", type=str, help="Output filename template")
parser.add_argument('-g', '--gpu', action='store_true', help="Use GPU for computations")

args = parser.parse_args()

if not args.gpu:
    print '=============Ignoring GPU============='
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

configuration = check_test_config(args.config, verbose=False)
if configuration is None:
    exit(0)

test(configuration, args.generator)
