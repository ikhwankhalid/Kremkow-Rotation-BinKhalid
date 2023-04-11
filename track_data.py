import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import scipy.signal as signal
import os
from utils import min2s, min_time

# parameters
raw_file = 'continuous.dat'
n_ch = 8
short_window = 100
sampling_freq = 30000.
state_colors = ['r', 'k', 'm', 'g']
ylim = [-1000, 18000]

# load data
mm = np.memmap(raw_file, dtype=np.int16, mode='r')
mm = mm.reshape((n_ch, -1), order='F')

# Get angle, linear displacement, time
angle = mm[4, :].copy().astype('float')     # angle
linear = mm[3, :].copy().astype('float')    # linear track
time = np.arange(angle.shape[0])
time_s = time / sampling_freq               # seconds

# Use shortened arrays to save memory and computation time
angle_short = angle[0::short_window]
linear_short = linear[0::short_window]
time_s_short = time_s[0::short_window]

# for experiment == '2022-04-06_16-04-19':
# we need to remove first part because it is when the platform signals is
# still not ok
start_experiment = 906
stop_experiment = 3250

idx_use = (time_s_short > start_experiment) & (
    time_s_short < stop_experiment)

angle_short = angle_short[idx_use]
time_s_short = time_s_short[idx_use]
linear_short = linear_short[idx_use]

# smooth the data
angle_smooth_short = ndimage.gaussian_filter1d(angle_short, 100.)
linear_smooth_short = ndimage.gaussian_filter1d(linear_short, 100.)

# sort angles into histogram bins to find peaks
hs_range = np.linspace(ylim[0], ylim[1], 100)
hs = np.histogram(angle_smooth_short, hs_range)
hist = hs[0]
hist = hist / hist.max()
peaks = signal.find_peaks(hist)
peaks = peaks[0]
peaks_x = hist[peaks]
peaks_y = hs_range[0:-1][peaks]

# the 4 top
sort_idx = np.argsort(peaks_x)
best_peaks = peaks[sort_idx[-4::]]
peak_values = hs_range[0:-1][best_peaks]
print(peak_values)
peak_hist = hist[best_peaks]

# calc threhsolds
peak_values_copy = peak_values.copy()
peak_values_copy.sort()

mean_state_values = peak_values_copy

state_values = np.ones([angle_smooth_short.shape[0], 4])

state_values[:, 0] *= mean_state_values[0]
state_values[:, 1] *= mean_state_values[1]
state_values[:, 2] *= mean_state_values[2]
state_values[:, 3] *= mean_state_values[3]

state_values[:, 0] = np.abs(state_values[:, 0]-angle_smooth_short)
state_values[:, 1] = np.abs(state_values[:, 1]-angle_smooth_short)
state_values[:, 2] = np.abs(state_values[:, 2]-angle_smooth_short)
state_values[:, 3] = np.abs(state_values[:, 3]-angle_smooth_short)

states = np.argmin(state_values, axis=1)
states += 1

start = time_s[0]
stop = time_s[-1]

plt.figure(1)
plt.clf()

plt.plot(time_s_short, states)

plt.yticks([1, 2, 3, 4], ['S4', 'S3', 'S2', 'S1'])
plt.ylim([0, 5])


plt.xlim([start, stop])

plt.savefig(os.path.join('plots', 'overview.png'), dpi=300)
plt.close()

offs = 6


def normalise(array):
    return (array - np.mean(array)) / np.std(array)


linear_normed = normalise(linear_smooth_short)
angle_normed = normalise(angle_smooth_short) + offs
states_normed = normalise(states) + offs


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(25,20))


ax1.plot(time_s_short, linear_normed)
ax1.plot(time_s_short, states_normed, color='red')
ax1.scatter(time_s_short, angle_normed, color='orange')
# plt.savefig(os.path.join('plots', "smooth.png"))
# plt.close()

print(linear_normed[-1])
filter_idxs = linear_normed < -0.30
linear_normed_filtered = linear_normed[filter_idxs]

# plt.figure(figsize=(20, 10))
ax2.plot(time_s_short[filter_idxs], linear_normed_filtered)
ax2.plot(time_s_short, states_normed - offs / 2, color='red')
ax2.scatter(time_s_short[filter_idxs], angle_normed[filter_idxs] - offs / 2, color='orange')
# plt.plot(angle_smooth_short[200000:400000])
plt.savefig(os.path.join('plots', "smooth_filtered.png"))
plt.close()

# diffmax = np.diff(angle_smooth_short).max()
# print(diffmax)

# data = mm[1, :].copy().astype('float')[::short_window]
# plt.figure()
# plt.plot(time_s[::short_window], data)

# times = [(7, 12), ]
# idx_min = min_time(time_s[::short_window], 7, 12)
# plt.vlines(
#     idx_min, data.min(), data.max(), color='red', linewidth=10, zorder=100
# )

# plt.savefig(os.path.join('plots', "ch_1.png"))
