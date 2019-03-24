import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Test block')
parser.add_argument("experiment", type=str, help="Name of the experiment")

args = parser.parse_args()

import os

from configuration_info.filestructure import get_data_name
from environment.environment_aux import collect_filenames, to_ts
from feature.extractor import iterate_dataset, PacketFeaturer


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


start_time = None
ground_time = None
requests = []
featurer = PacketFeaturer(None, True)
data = []
time = []

limits = [
    [(0, 10), (0, 1)],
    [(0, 10), (0, 1)],
    [(0, 0.2), (0, 1)],
    [(0, 10), (0, 1)],
    [(0, 100000), (0, 1)],
    [(0, 100000), (0, 1)],
    [(0, 40), (0, 1)],
[(0, 100000), (0, 1)],
[(0, 40), (0, 1)],
[(0, 100000), (0, 1)],
[(0, 100000), (0, 1)],
[(0, 0.5), (0, 1)],
[(0, 15), (0, 1)],
[(0, 40), (0, 1)],
[(0, 1.5e8), (0, 1)],
[(0, 40), (0, 1)],
[(0, 40), (0, 1)],
]

last = 0
sk_sz = 100000 - 1

print collect_filenames(get_data_name(args.experiment))[0]

if not os.path.exists(args.experiment):
    os.makedirs(args.experiment)

output_data_file = open(args.experiment + '/data.csv', 'w')
header = ','.join(['timestamp'] + featurer.names) + '\n'
output_data_file.write(header)

lts = None

for request in iterate_dataset(collect_filenames(get_data_name(args.experiment))):

    requests.append(request)
    lts = request['timestamp']

    if sk_sz > 0:
        sk_sz -= 1
        featurer.logical_time += 1
        featurer.real_time = request['timestamp']
        featurer.preserve()
        continue

    if sk_sz == 0:
        requests = []
        sk_sz -= 1

    if start_time is None:
        start_time = request['timestamp']
        ground_time = start_time

    if request['timestamp'] - start_time >= 60:
        features = featurer.gen_feature_set(requests, pure=True)
        featurer.preserve()

        mean = np.mean(features, axis=0).tolist()
        data_string = ','.join([str(lts)] + [str(item) for item in mean]) + '\n'
        output_data_file.write(data_string)
        output_data_file.flush()
        data.append(mean)
        time.append(request['timestamp'])

        print 'Collected:', len(requests), 'VS', to_ts(start_time - ground_time), 'Iter', len(data)

        requests = []
        start_time = request['timestamp']

features = featurer.gen_feature_set(requests, pure=True)
featurer.preserve()

mean = np.mean(features, axis=0).tolist()

data_string = ','.join([str(lts)] + [str(item) for item in mean]) + '\n'
output_data_file.write(data_string)
output_data_file.flush()
data.append(mean)

output_data_file.close()