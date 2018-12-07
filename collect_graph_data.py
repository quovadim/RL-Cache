import os
import argparse
from os import listdir
from os.path import isfile, join
import sys

parser = argparse.ArgumentParser(description='Algorithm tester')
parser.add_argument("networks", type=str, help="Network name suffix")
parser.add_argument('-r', "--region", type=str, default='american', help="Data region")
parser.add_argument('-l', '--length', type=int, default=4000, help="Trace length")
parser.add_argument('-p', '--period', type=int, default=100, help="Dump period")
parser.add_argument('-g', '--gpu', action='store_true', help="Use GPU for computations")
parser.add_argument('-e', '--eviction', action='store_true', help="Predict eviction model")
parser.add_argument('-a', '--admission', action='store_true', help="Predict admission model")
parser.add_argument('-s', '--secondhit', action='store_true', help="Predict second hit")

args = parser.parse_args()

if not args.gpu:
    print '=============Ignoring GPU============='
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras

from GameEnvironment import GameEnvironment

filepath = 'data/' + args.region + '_featured/'
filenames = [(join(filepath, f), int(f.replace('.csv', ''))) for f in listdir(filepath) if isfile(join(filepath, f)) and '.csv' in f and 'lock' not in f]
filenames = sorted(filenames, key=lambda x: x[1])
filenames = [item[0] for item in filenames]

iterations = 1000 * args.length
env = GameEnvironment(filepath, 0, skip=0)

if args.admission:
    env.model_admission.load_weights('models/adm_' + args.networks)
if args.eviction:
    env.model.load_weights('models/evc_' + args.networks)

rows = []

counter = 0
tcounter = 0

if args.secondhit:
    features_file = open('auxiliary/features', 'w')
    predictions_sh = open('auxiliary/sh_predictions', 'w')
if args.admission:
    admission_predictions_file = open('auxiliary/admission_predictions_' + args.networks, 'w')
if args.eviction:
    eviction_predictions_file = open('auxiliary/eviction_predictions_' + args.networks, 'w')

skip_rows = 2000000

for row in GameEnvironment.iterate_dataset_over_all(filenames, 0):
    if skip_rows > 0:
        skip_rows -= 1
        env.featurer.update_packet_state(row)
        env.featurer.update_packet_info(row)
        continue
    if skip_rows == 0:
        env.featurer.preserve()
        skip_rows -= 1

    rows.append(row)

    if (1 + tcounter) % 1000 == 0:
        sys.stdout.write('\rIteration : ' + str(1 + tcounter))
        sys.stdout.flush()

    if len(rows) == args.period * 1000:
        print ''
        env.featurer.preserve()
        if args.secondhit:
            print 'Collecting classical features'
            classical_feature_set = env.gen_feature_set(rows, classical=True, pure=True, verbose=True).tolist()
            print 'Collecting classical predictions'
            predictable_feature_set = env.gen_feature_set(rows, classical=True, pure=False, verbose=True).tolist()
        if args.eviction or args.admission:
            print 'Collecting ML features'
            predictable_feature_set_ml = env.gen_feature_set(rows, classical=False, verbose=True)

            if args.admission:
                print 'Prediction admission'
                predictions_admission = env.model_admission.predict(
                    predictable_feature_set_ml, batch_size=4096, verbose=1).tolist()
            if args.eviction:
                print 'Predicting eviction'
                predictions_eviction = env.model.predict(
                    predictable_feature_set_ml, batch_size=4096, verbose=1).tolist()

        for i in range(len(rows)):
            if args.secondhit:
                features_file.write(' '.join([str(item) for item in classical_feature_set[i]]) + '\n')
                predictions_sh.write(' '.join([str(item) for item in predictable_feature_set[i]]) + '\n')

            if args.admission:
                admission_predictions_file.write(' '.join([str(item) for item in predictions_admission[i]]) + '\n')
            if args.eviction:
                eviction_predictions_file.write(' '.join([str(item) for item in predictions_eviction[i]]) + '\n')
        counter += len(rows)
        rows = []

    if counter >= iterations:
        break

    tcounter += 1
