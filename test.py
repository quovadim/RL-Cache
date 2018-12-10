import argparse
import os
from os import listdir
from os.path import isfile, join
import json

from GameEnvironment import GameEnvironment

parser = argparse.ArgumentParser(description='Algorithm trainer')
parser.add_argument("networks", type=str, help="Network name suffix")
parser.add_argument("config", type=str, help="Configuration file for training")
parser.add_argument("generator", type=str, help="Output filename template")
parser.add_argument('-g', '--gpu', action='store_true', help="Use GPU for computations")
parser.add_argument('-e', '--preload_eviction', action='store_true', help="Load pretrained eviction")
parser.add_argument('-a', '--preload_admission', action='store_true', help="Load pretrained admission")

args = parser.parse_args()

configuration = json.load(open(args.config, 'r'))

if not args.gpu:
    print '=============Ignoring GPU============='
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

filepath = configuration['data folder']
filenames = [(join(filepath, f), int(f.replace('.csv', ''))) for f in listdir(filepath)
             if isfile(join(filepath, f)) and '.csv' in f and 'lock' not in f]
filenames = sorted(filenames, key=lambda x: x[1])
filenames = [item[0] for item in filenames]

env = GameEnvironment(configuration)

if args.preload_eviction:
    print 'Loading pretrained from', 'models/evc_' + args.networks
    env.model_eviction.load_weights('models/evc_' + args.networks)
if args.preload_admission:
    print 'Loading pretrained from', 'models/adm_' + args.networks
    env.model_admission.load_weights('models/adm_' + args.networks)

env.test(filenames, args.generator)
