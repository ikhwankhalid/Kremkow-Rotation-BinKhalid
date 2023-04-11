#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 05:04:55 2022

@author: kailun
"""

# %%
import pandas
import numpy as np
import matplotlib.pyplot as plt
import os
# import cv2 as cv2
import scipy.ndimage as ndimage
import numpy.ma as ma
from scipy import interpolate
# import pycorrelate as pyc
from scipy.signal import butter, filtfilt
from scipy import interpolate
from scipy.interpolate import CubicSpline
from matplotlib import rcParams
import scipy.signal as signal

# %%

experiment = '2022-04-06_16-04-19'
raw_file = 'continuous.dat'

experiment = '2022-04-06_16-04-19'
# "D:\2022-04-06_16-04-19_Record Node 105\experiment1\recording1\continuous\NI-DAQmx-103.0"
raw_file = 'F:/2022JanBatch_TL/mouse_3_all/2022-04-06_16-04-19/Record Node 105/experiment1/recording1/continuous/NI-DAQmx-103.0/continuous.dat'
start_experiment = 906
stop_experiment = 3250


n_ch = 8
mm = np.memmap(raw_file, dtype=np.int16, mode='r')
mm = mm.reshape((n_ch, -1), order='F')

# %%

# %%
angle = mm[4, :].copy().astype('float')  # angle

# %%
time = np.arange(angle.shape[0])
time_s = time / 30000.

angle_short = angle[0::100]
time_s_short = time_s[0::100]

if experiment == '2022-04-06_16-04-19':
    # we need to remove first part because it is when the platform signals is still not ok
    idx_use = (time_s_short > start_experiment) & (
        time_s_short < stop_experiment)

    angle_short = angle_short[idx_use]
    time_s_short = time_s_short[idx_use]
# %%
angle_smooth_short = ndimage.gaussian_filter1d(angle_short, 100.)


# %%
state_colors = ['r', 'k', 'm', 'g']
#
plt.figure(1)
plt.clf()

ylim = [-1000, 18000]

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
# states_smooth = ndimage.gaussian_filter1d(states,100.)

thr_s4s3 = peak_values_copy[0] + (peak_values_copy[1]-peak_values_copy[0])/2
thr_s3s2 = peak_values_copy[1] + (peak_values_copy[2]-peak_values_copy[1])/2
thr_s2s1 = peak_values_copy[2] + (peak_values_copy[3]-peak_values_copy[2])/2

angle_smooth_short_gradient_abs = np.abs(np.gradient(angle_smooth_short))

thresholds = [thr_s4s3, thr_s3s2, thr_s2s1]
#
peak_thr = angle_smooth_short_gradient_abs.max()*1/3
peaks = signal.find_peaks(
    angle_smooth_short_gradient_abs, height=peak_thr, distance=100)
peaks_idx = peaks[0]


plt.axes([0.05, 0.2, 0.1, 0.7])
plt.plot(hist, hs_range[0:-1])

plt.plot(peak_hist, peak_values, '.')

plt.plot([0, 1], [thr_s4s3, thr_s4s3], 'r')
plt.plot([0, 1], [thr_s3s2, thr_s3s2], 'k')
plt.plot([0, 1], [thr_s2s1, thr_s2s1], 'm')


plt.ylim(ylim)

ax = plt.axes([0.25, 0.65, 0.7, 0.3])
plt.plot(time_s_short, angle_short)

plt.plot(time_s_short, angle_smooth_short)

plt.plot(time_s_short[peaks_idx], angle_smooth_short[peaks_idx], 'o')

# plt.plot(time_s[0::100],angle_smooth_interp[0::100])


# plt.plot(time_s[0::100],angle_smooth[no_transient_noise][0::100])

start = time_s[0]
stop = time_s[-1]
# plt.plot([start,stop],[thr_s4s3,thr_s4s3],'r')
# plt.plot([start,stop],[thr_s3s2,thr_s3s2],'k')
# plt.plot([start,stop],[thr_s2s1,thr_s2s1],'m')

for i in range(4):
    pt = peak_values[i]
    plt.plot([start, stop], [pt, pt], color=state_colors[i])

plt.ylim(ylim)


ax = plt.axes([0.25, 0.35, 0.7, 0.2], sharex=ax)
plt.plot(time_s_short, angle_smooth_short_gradient_abs)
plt.plot(time_s_short[peaks_idx],
         angle_smooth_short_gradient_abs[peaks_idx], 'o')


plt.axes([0.25, 0.1, 0.7, 0.2], sharex=ax)
plt.plot(time_s_short, states)
# plt.plot(time_s[no_transient_noise][0::100],states_smooth[no_transient_noise][0::100])

plt.yticks([1, 2, 3, 4], ['S4', 'S3', 'S2', 'S1'])
plt.ylim([0, 5])


plt.xlim([start, stop])

plt.savefig(experiment+'_overview.png', dpi=300)
#

# %% detect transitions

# %%
sampling_rate = 300
window_mta = np.int(1.*sampling_rate)

window_index = np.arange(-window_mta, window_mta, 1)
window_s = window_index/sampling_rate


transitions = np.zeros([len(window_s), len(peaks_idx)])
transitions_smooth = np.zeros([len(window_s), len(peaks_idx)])

state_transitions = np.zeros(peaks_idx.shape)
count = 0
for i, index_stamp in enumerate(peaks_idx):
    index_stamp = int(index_stamp)
    # idx = np.searchsorted(timestamps_NP_s, time_stamp)
    idxs = index_stamp+window_index
    # up_mov_time[:,i] = timestamps_NP_s[idxs]
    if idxs.min() <= 0:
        continue
    if idxs.max() > len(angle):
        continue
    transitions[:, count] = angle_short[idxs]
    transitions_smooth[:, count] = angle_smooth_short[idxs]

    # we identify the states
    tmp_states = states[idxs].copy()
    state_before = tmp_states[0:10].mean()
    state_after = tmp_states[-10::].mean()

    state_transitions[i] = 10*state_before + state_after

    # we characterize
    # inital state

    # left_movement_time_for_MTA_spikes.append(left_movement_time[i])
    count += 1


fig = plt.figure(9)
plt.clf()

peak_values.sort()

count = 1
ax = fig.subplots(4, 4)  # ([4, 4])
for state_before in range(4):
    for state_after in range(4):
        ax_tmp = ax[state_before, state_after]
        plt.axes(ax_tmp)

        if state_before == state_after:
            ax_tmp.axis('off')
            continue

        plt.title(str(state_before+1)+' -> '+str(state_after+1))

        current_state = 10*(state_before+1) + state_after+1

        # plt.subplot(4,4,count)
        idx_state = state_transitions == current_state

        plt.plot(transitions_smooth[:, idx_state])

        x1 = 0
        x2 = transitions_smooth.shape[0]
#        plt.plot([0,x2],[thr_s4s3,thr_s4s3],'r')
#        plt.plot([0,x2],[thr_s3s2,thr_s3s2],'k')
#        plt.plot([0,x2],[thr_s2s1,thr_s2s1],'m')
#
        for i in range(4):
            pt = peak_values[i]
            plt.plot([0, x2], [pt, pt], color=state_colors[i])

        plt.ylim([-1000, 18000])
        plt.yticks(peak_values, ['S1', 'S2', 'S3', 'S4'])

        count += 1


fig.tight_layout()

plt.savefig(experiment+'_detected_state_transitions.png', dpi=300)
#
# all_threshold_crossings = []
# for thr_value in thresholds:
#    #thr_value = thr_s1
#    # up
#    thr_idx, = np.where(angle_smooth_short >= thr_value)
#    gaps, = np.where(np.diff(thr_idx)> 30)
#    thr_idx_upwards = thr_idx[gaps+1]
#    if thr_idx[0] > 0:
#        thr_idx_upwards = np.append(thr_idx[0],thr_idx_upwards)
#
#    # down
#    thr_idx, = np.where(angle_smooth_short <= thr_value)
#    gaps, = np.where(np.diff(thr_idx)> 30)
#    thr_idx_downwards = thr_idx[gaps+1]
#    if thr_idx[0] > 0:
#        thr_idx_downwards = np.append(thr_idx[0],thr_idx_downwards)
#
#
#    thr_idx_new = np.append(thr_idx_upwards,thr_idx_downwards)
#    thr_idx_new.sort()
#
#
#    plt.plot([0,len(angle_smooth_short)],[thr_value,thr_value])
#    plt.plot(thr_idx_new,np.ones(thr_idx_new.shape)*thr_value,'o')
#
#    all_threshold_crossings = np.append(all_threshold_crossings,thr_idx_new)
#
# all_threshold_crossings.sort()

# now we have to filter the crossing and only take the firts


# %%

# %%

def getTurningDirections(platformStates, ignoreInvalid=False):
    """To get the turning directions given platform states.
    PARAMETERS
    ----------
    platformStates : list or array_like, 1D
        The platform states (1, 2, 3, 4) for N time points.

    RETURN
    ------
    turningDirections : list, 1D
        The corresponding direction of turning for the N time points (length = N - 1).
        0: no turning; 
        -1: left/anticlockwise turn; 
        1: right/clockwise turn.
    """
    turningDirections = []
    leftTurns = [(1, 2), (2, 4), (3, 1), (4, 3)]
    rightTurns = [(1, 3), (2, 1), (3, 4), (4, 2)]
    for i, curState in enumerate(platformStates[1:], 1):
        prevState = platformStates[i-1]
        if curState == prevState:
            turningDirections.append(0)
        elif (prevState, curState) in leftTurns:
            turningDirections.append(-1)
        elif (prevState, curState) in rightTurns:
            turningDirections.append(1)
        else:
            if ignoreInvalid:
                turningDirections.append(np.nan)
            else:
                raise ValueError(
                    f"Invalid turning directions: state {prevState} to state {curState} at time points ({i-1},{i})!")
    return turningDirections
