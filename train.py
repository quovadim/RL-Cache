import argparse

parser = argparse.ArgumentParser(description='Algorithm trainer')
parser.add_argument("experiment", type=str, help="Name of the experiment")
parser.add_argument('-t', '--threads', type=int, default=10, help="Number of threads")
parser.add_argument('-c', '--cpu', action='store_true', help="Use CPU for computations")
parser.add_argument('-e', '--preload_eviction', action='store_true', help="Load pretrained eviction")
parser.add_argument('-a', '--preload_admission', action='store_true', help="Load pretrained admission")
parser.add_argument('-v', '--verbose', action='store_true', help="Verbose sanity check")
parser.add_argument('-s', '--show', action='store_true', help="Show testing results")

args = parser.parse_args()

import os

from environment.environment import train
from configuration_info.config_sanity import check_train_config

configuration = check_train_config(args.experiment, verbose=False)
if configuration is None:
    exit(0)

if args.cpu:
    print '=============Ignoring GPU============='
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

train(configuration, args.preload_admission, args.preload_eviction,
      n_threads=args.threads, verbose=args.verbose, show=not args.show)

