from os import listdir
from os.path import isfile, join
import argparse
import sys

parser = argparse.ArgumentParser(description='Data collector')
parser.add_argument("filepath", type=str, help="File path")
parser.add_argument("output", type=str, help="Output filename")
parser.add_argument('-k', '--skip', type=int, default=0, help="Skip files")
parser.add_argument('-i', '--iterations', type=int, default=1000, help="Number of result lines")

args = parser.parse_args()

filenames = sorted([join(args.filepath, f) for f in listdir(args.filepath)
             if isfile(join(args.filepath, f)) and '.out' in f])

filenames = filenames[args.skip:]

result_file = open(args.output, 'w')

counter = 0
latest_timestamp = 0
for fname in filenames:
    ifile = open(fname, 'r')
    for line in ifile.readlines():
        if counter > args.iterations * 1000:
            break
        if counter % 1000 == 0:
            sys.stdout.write('\r' + str(counter))
            sys.stdout.flush()
            result_file.flush()
        timestamp = int(line.split(' ')[0])
        if timestamp < latest_timestamp:
            continue
        timestamp = latest_timestamp
        result_file.write(line)
        counter += 1
    ifile.close()
    if counter > args.iterations * 1000:
        break

print ''
result_file.close()