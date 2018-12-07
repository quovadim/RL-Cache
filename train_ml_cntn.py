import argparse
import os
from os import listdir
from os.path import isfile, join
import numpy as np

parser = argparse.ArgumentParser(description='Algorithm trainer')
parser.add_argument("networks", type=str, help="Network name suffix")
parser.add_argument('-s', '--size', type=int, default=1228, help="Cache size")
parser.add_argument('-r', "--region", type=str, default='china', help="Data region")
parser.add_argument('-t', '--threads', type=int, default=10, help="Number of threads")
parser.add_argument('-i', '--iterations', type=int, default=50, help="Number of iterations per epoch")
parser.add_argument('-l', '--length', type=int, default=300, help="Trace length")
parser.add_argument('-c', '--cpu', action='store_true', help="Use CPU for computations")
parser.add_argument('-p', '--preload', action='store_true', help="Load pretrained models")

args = parser.parse_args()

#np.random.seed(33)

if args.cpu:
    print '=============Ignoring GPU============='
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

filepath = 'data/' + args.region + '_featured/'
filenames = [(join(filepath, f), int(f.replace('.csv', ''))) for f in listdir(filepath)
             if isfile(join(filepath, f)) and '.csv' in f and 'lock' not in f]
filenames = sorted(filenames, key=lambda x: x[1])
filenames = [item[0] for item in filenames]

from GameEnvironment import GameEnvironment

cache_size = args.size
iterations = 1000 * args.length
env = GameEnvironment(filepath, cache_size * 1024 * 1024, args.networks, skip=0)

if args.preload:
    print 'Loading pretrained from', 'models/evc_' + args.networks
    env.model.load_weights('models/evc_' + args.networks)
if args.preload:
    print 'Loading pretrained from', 'models/adm_' + args.networks
    env.model_admission.load_weights('models/adm_' + args.networks)

n_threads = args.threads

env.run_and_train_ml(args.iterations, 100, iterations, 1000000, filenames, 0, n_threads=args.threads)
