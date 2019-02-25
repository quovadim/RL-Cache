import os
import argparse

parser = argparse.ArgumentParser(description='Concatenate for optimal')
parser.add_argument("folder", type=str, help="Input")
parser.add_argument("l", type=int, help="Min filenum")
parser.add_argument("u", type=int, help="Max filenum")
parser.add_argument("output", type=str, help="Output")

args = parser.parse_args()

file_list = [args.folder + str(i) + '.csv' for i in range(args.l, args.u)]
cmd_list = ' '.join(file_list)

command = 'cat ' + cmd_list + ' > ' + args.output
os.system(command)