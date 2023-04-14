import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import scipy.signal as signal
import os
from utils import min2s, min_time, getVideoInfo
import settings
import datetime
from process_vids import cut_video

# directories
exp_dir = os.path.join(
    settings.proj_dir,
    'data',
    '2022-04-06_16-04-19_Record Node 105',
    'experiment1'
)
analog_dir = os.path.join(
    settings.proj_dir,
    exp_dir,
    'recording1',
    'continuous',
    'NI-DAQmx-103.0'
)
ttl_dir = os.path.join(
    settings.proj_dir, 
    exp_dir, 
    'recording1', 
    'events', 
    'NI-DAQmx-103.0', 
    'TTL_1'
)

# parameters
n_ch = 8
short_window = 100
sampling_freq = 30000.
state_colors = ['r', 'k', 'm', 'g']
ylim = [-1000, 18000]
start_ttl = 89113001
ttlbreakthresh = 400

# Load data
# Analog data containing linear track movement
mm = np.memmap(
    os.path.join(analog_dir, 'continuous.dat'), dtype=np.int16, mode='r'
)
mm = mm.reshape((n_ch, -1), order='F')

# TTL data to get when cameras are active
channel_states = np.load(
    os.path.join(ttl_dir, 'channel_states.npy'), allow_pickle=True
)
timestamps_ttl = np.load(
    os.path.join(ttl_dir, 'timestamps.npy'), allow_pickle=True
)

# channel_states =+ np.amin(channel_states)
mask = channel_states == 2.
ch1_times = timestamps_ttl[mask]
times_ttl = ch1_times - start_ttl
video_dat = getVideoInfo(times_ttl, ttlbreakthresh, sampling_freq)
print(video_dat[4])

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
start_experiment = (video_dat[2][1] + sampling_freq) / sampling_freq
stop_experiment = (video_dat[3][2] - sampling_freq) / sampling_freq

idx_use = (time_s_short > start_experiment) & (
    time_s_short < stop_experiment)


# angle_short[~idx_use] = angle_short[~idx_use] + 100
# linear_short[~idx_use] = linear_short[~idx_use] + 100

# angle_short = angle_short - tempo
# linear_short = linear_short - tempo

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

# start = time_s[0]
# stop = time_s[-1]


def normalise(array):
    array -= np.amin(array)
    return array / np.amax(array)


linear_normed = normalise(linear_smooth_short)
angle_normed = normalise(angle_smooth_short)
states_normed = normalise(states)


filter_idxs = linear_normed < 0.1
linear_normed_filtered = linear_normed[filter_idxs]
time_s_short_filtered = time_s_short[filter_idxs]


def cut_from_filtered(
    track_filtered,
    time_filtered,
    vid_dat,
    vid_idx,
    sampling_freq,
):
    """_summary_

    Args:
        track_filtered (_type_): _description_
        time_filtered (_type_): _description_
        vid_dat (_type_): _description_
        vid_idx (_type_): _description_
        sampling_freq (_type_): _description_

    Returns:
        _type_: _description_
    """
    idx_start = np.argmin(
        np.abs(time_filtered - vid_dat[2][vid_idx] / sampling_freq)
    )
    idx_end = np.argmin(
        np.abs(time_filtered - vid_dat[3][vid_idx] / sampling_freq)
    )
    track_out = track_filtered[idx_start:idx_end]
    time_out = time_filtered[idx_start:idx_end]

    t_diffs = np.diff(time_out)
    timeBreakThresh = np.amin(t_diffs) * 1.5
    t_data = getVideoInfo(time_out, timeBreakThresh, 1.)

    return track_out, time_out, t_data


track1, time1, t1_data = cut_from_filtered(
    linear_normed_filtered, time_s_short_filtered, video_dat, 1, sampling_freq
)


begin_times = np.round(t1_data[2] - time1[0], 0)
end_times = np.round(t1_data[3] - time1[0], 0)
durations = np.round(t1_data[4], 0)
# cut_times = list(zip(begin_times, end_times))
cut_times = list(zip(begin_times, durations))
# print(cut_times)
for entry in video_dat[4]:
    print(str(datetime.timedelta(seconds=entry)))


# Minivut video
"""for cut_time in cut_times:
    start_vid = str(datetime.timedelta(seconds=cut_time[0]))
    vid_duration = str(datetime.timedelta(seconds=cut_time[1]))
    vid_in = os.path.join(settings.proc_dir, 'cam4_2022-04-06-16-20-35.mp4')
    vid_out = os.path.join(
        settings.mini_dir, f'cam4_2022-04-06-16-20-35_{cut_time[0]}.mp4'
    )
    os.makedirs(settings.mini_dir, exist_ok=True)
    cut_video(vid_in, vid_out, start_vid, vid_duration)"""

###############################################################################
# Plotting                                                                    #
###############################################################################
# Overview plot
plt.figure(figsize=(25, 5))
plt.plot(time_s_short, linear_normed)
plt.plot(time_s_short, states_normed, color='red')
plt.scatter(time_s_short, angle_normed, color='orange')
plt.vlines(
    np.hstack((video_dat[2], video_dat[3])) / sampling_freq,
    -0.1,
    1,
    color='m',
    linewidth=2.
)
plt.vlines(
    t1_data[2],
    -0.1,
    1,
    color='green',
    linewidth=0.5
)
plt.vlines(
    t1_data[3],
    -0.1,
    1,
    color='green',
    linewidth=0.5,
    linestyle="--"
)
plt.xlim(t1_data[2][0] - 20, t1_data[3][-1] + 20)
plt.savefig(os.path.join('plots', "moveit.png"))
plt.close()


# Filter linear track movement
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(25, 20))
offs = 2

ax1.plot(time_s_short, linear_normed)
ax1.plot(time_s_short, states_normed + offs, color='red')
ax1.scatter(time_s_short, angle_normed + offs, color='orange')

ax2.plot(time_s_short[filter_idxs], linear_normed_filtered)
ax2.plot(time_s_short, states_normed + offs - 1., color='red')
ax2.scatter(
    time_s_short[filter_idxs],
    angle_normed[filter_idxs] + offs - 1.,
    color='orange'
)
plt.savefig(os.path.join('plots', "smooth_filtered.png"))
plt.close()


# Plot when cameras are active(?)
plt.figure(figsize=(15, 5))
plt.scatter(ch1_times, np.ones(ch1_times.shape), s=.01)
plt.savefig(os.path.join('plots', "channel_times.png"))
plt.close()
