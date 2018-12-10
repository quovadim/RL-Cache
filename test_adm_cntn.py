import os
import argparse
from os import listdir
from os.path import isfile, join

parser = argparse.ArgumentParser(description='Algorithm tester')
parser.add_argument("filename", type=str, help="Output filename")
parser.add_argument("networks", type=str, help="Network name suffix")
parser.add_argument('-r', "--region", type=str, default='american', help="Data region")
parser.add_argument('-s', '--size', type=int, default=1228, help="Cache size")
parser.add_argument('-l', '--length', type=int, default=400, help="Trace length")
parser.add_argument('-p', '--period', type=int, default=100, help="Dump period")
parser.add_argument('-g', '--gpu', action='store_true', help="Use GPU for computations")

args = parser.parse_args()

if not args.gpu:
    print '=============Ignoring GPU============='
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras

from GameEnvironment import GameEnvironment

filepath = 'data/' + args.region + '_featured/'
filenames = [(join(filepath, f), int(f.replace('.csv', ''))) for f in listdir(filepath)
             if isfile(join(filepath, f)) and '.csv' in f and 'lock' not in f]
filenames = sorted(filenames, key=lambda x: x[1])
filenames = [item[0] for item in filenames]

cache_size = args.size
iterations = 1000 * args.length
env = GameEnvironment(filepath, cache_size * 1024 * 1024, skip=0)

if True:
    print 'Loading pretrained from', 'models/adm_' + args.networks
    env.model_admission.load_weights('models/adm_' + args.networks)

env.test_adm_infinite(iterations, args.period, filenames, 0, args.filename)
