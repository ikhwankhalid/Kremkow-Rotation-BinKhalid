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
    'nidaq_2022-12-19_18-53-46',
    'experiment1'
)
analog_dir = os.path.join(
    settings.proj_dir,
    exp_dir,
    'recording1',
    'continuous',
    'NI-DAQmx-100.0'
)
ttl_dir = os.path.join(
    settings.proj_dir,
    exp_dir, 
    'recording1',
    'events',
    'NI-DAQmx-100.0',
    'TTL_1'
)

# parameters
n_ch = 8
short_window = 100
sampling_freq = 30000.
state_colors = ['r', 'k', 'm', 'g']
ylim = [-1000, 18000]
start_ttl = 676008001
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

mask = channel_states == 2.
ch1_times = timestamps_ttl[mask]
times_ttl = ch1_times - start_ttl
video_dat = getVideoInfo(times_ttl, ttlbreakthresh, sampling_freq)
print(video_dat[4])
