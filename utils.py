import numpy as np

def min2s(min, s):
    return min * 60 + s


def min_time(time, min, s):
    return time[np.argmin(np.abs(time - min2s(min, s)))]
