import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


parser = argparse.ArgumentParser(description='Algorithm tester')

parser.add_argument("-f", "--filename", type=str, default='0', help="Output filename")

parser.add_argument('-s', '--skip', type=int, default=0, help="Skip")
parser.add_argument('-m', '--smooth', type=int, default=1, help="Smooth factor")
parser.add_argument('-x', '--extension', type=str, default='pdf', help="Target extension")

parser.add_argument('-e', '--remove', action='store_true', help="Remove previous graphs")

parser.add_argument('-l', '--plots', action='store_true', help="Build sizes graphs")
parser.add_argument('-p', '--percentiles', action='store_true', help="Build size-aware percentiles")

args = parser.parse_args()

algorithms = ['GDSF', 'LRU', 'LFU']
regions = ['china', 'usa']

command = 'build_algorithm_description.py'

configs_source_dir = 'configs/'
output_source_dir = 'graphs/'

args_to_add = []
if args.percentiles:
    args_to_add.append('-p')

if args.plots:
    args_to_add.append('-l')

if args.remove:
    args_to_add.append('-e')

args_to_add.append('-x=' + str(args.extension))
args_to_add.append('-m=' + str(args.smooth))
args_to_add.append('-s=' + str(args.skip))
args_to_add.append('-f=' + str(args.filename))

for algorithm in algorithms:
    for region in regions:
        config_file = configs_source_dir + 'M' + algorithm + '_test_' + region + '.json'
        output_dir = output_source_dir + 'M' + algorithm + '_' + region + '/'
        command_list = ['python', command, config_file, output_dir] + args_to_add
        print 'Executing', ' '.join(command_list)
        os.system(' '.join(command_list))
