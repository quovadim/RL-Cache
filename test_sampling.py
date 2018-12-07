import numpy as np


def sample(distribution):
    target = np.random.uniform(0, 1)
    summary = 0
    for i in range(len(distribution)):
        summary += distribution[i]
        if summary > target:
            return i
    return len(distribution) - 1

v1 = [0.1, 0.9]

samples = [sample(v1) for _ in range(100000)]

u, c = np.unique(samples, return_counts=True)

print c * 1. / len(samples)