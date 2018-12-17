import matplotlib.pyplot as plt
import argparse
import numpy as np
import re

RUN_PATTERN = r"\A['R']['U']['N']\s\d*"
W_PATTERN = r"[W]\s\d+(\s\D+\s\d+[.]\d+)+"
B_PATTERN = r"[B]\s\d+(\s\D+\s\d+[.]\d+)+"
A_PATTERN = r"[A]\d+\s\d+(\s\D+\s\d+[.]\d+)+"


def parse_performance_line(data):
    data = data.strip()
    data = data.split(' ')
    prefix = data[0]
    data = data[1:]
    moment = int(data[0])
    data = data[1:]
    assert len(data) % 2 == 0
    perfs = {}
    for i in range(0, len(data), 2):
        perfs[data[i]] = float(data[i+1])
    return prefix, moment, perfs


def extract_performance(data):
    performance_data = [parse_performance_line(item) for item in data
                        if re.match(W_PATTERN, item)]
    return performance_data


def grab_points(data, keys):
    time_moments = []
    coloring = []
    seq_data = {}
    for key in keys:
        seq_data[key] = []
    color_index = 1
    for run in data:
        for p, moment, perf_data in run:
            time_moments.append(moment)
            coloring.append(color_index * 1.0 / len(data))
            for key in keys:
                seq_data[key].append(perf_data[key])
        color_index += 1
    return time_moments, seq_data, coloring


parser = argparse.ArgumentParser(description='Logs parser')
parser.add_argument("logs", type=str, help="Logs filename")

args = parser.parse_args()

data = open(args.logs, 'r').readlines()
data = [item.replace('\n', '') for item in data]

runs = []
counter = 0
for line in data:
    if re.match(RUN_PATTERN, line):
        runs.append(counter)
    counter += 1

#runs.append(len(data))

runs_aggregated = []
last_run = None
for i in range(1, len(runs)):
    left, right = runs[i-1], runs[i]
    run_id = int(data[left].replace('RUN ', ''))
    left = min(left+1, len(data) - 1)
    last_run = extract_performance(data[left:right])
    runs_aggregated.append(last_run)

time_moments_ml, seq_data_ml, coloring_ml = grab_points(runs_aggregated, ['ML-GDSF-DET'])
time_moments_dt, seq_data_dt, coloring_dt = grab_points([last_run], ['AL-GDSF'])

plt.scatter(time_moments_dt, seq_data_dt['AL-GDSF'], c='red', s=200, alpha=0.5)
plt.scatter(time_moments_ml, seq_data_ml['ML-GDSF-DET'], c=coloring_ml, cmap='jet', s=100, alpha=0.5)
plt.show()