import numpy as np


def fit_uniform_dist(xs):
    n = xs.shape[0]
    ranges = np.max(xs, axis=0) - np.min(xs, axis=0)
    lower = np.max(xs, axis=0) - ranges * (n + 2) / n
    upper = np.min(xs, axis=0) + ranges * (n + 2) / n
    return lower, upper