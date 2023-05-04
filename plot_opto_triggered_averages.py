#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 18:53:47 2023

@author: jenskremkow
"""
# %%
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
import pandas
from matplotlib import rcParams

# %%

# experiment_name = '2022-12-19_16-00-17'
experiment_name = '2022-12-19_18-53-46'
# experiment_name = '2022-12-20_15-49-59'
# experiment_name = '2022-12-20_16-51-54'
# experiment_name = '2022-12-22_13-52-57'
# experiment_name = '2022-12-22_16-24-43'

data_dir = 'data/' + experiment_name + '/'


fn_pupil = data_dir + 'pupil_dlc_data.npy'
fn_ttl_cont = data_dir + 'opto_ttl_cont.npy'

data_pupil = np.load(fn_pupil, encoding='latin1', allow_pickle=True).item()
data_opto = np.load(fn_ttl_cont, encoding='latin1', allow_pickle=True).item()


# %%
opto_right_analog = data_opto['ttl_opto_right']
opto_left_analog = data_opto['ttl_opto_left']

# %% event triggered average
opto_right_tmp = opto_right_analog
diff_ttl = np.diff(opto_right_tmp)
gaps_idx = np.where(diff_ttl > 50000)[0]

opto_right_starts = np.append(opto_right_tmp[0], opto_right_tmp[gaps_idx+1])

opto_left_tmp = opto_left_analog
diff_ttl = np.diff(opto_left_tmp)
gaps_idx = np.where(diff_ttl > 50000)[0]

opto_left_starts = np.append(opto_left_tmp[0], opto_left_tmp[gaps_idx+1])

if experiment_name == '2022-12-22_16-24-43':
    opto_left_starts = np.delete(opto_left_starts, 12)

timestamps_cont = data_opto['timestamps']
cont_opto_right = data_opto['cont_opto_right']
cont_opto_left = data_opto['cont_opto_left']

# %%
channel_states = np.load(data_dir+'TTL_1/channel_states.npy')
timestamps = np.load(data_dir+'TTL_1/timestamps.npy')

events = {}

for i in range(8):
    events[i] = timestamps[channel_states == i]


# %% video ttls

video_ttls_tmp = events[2]

diff_ttl = np.diff(video_ttls_tmp)
gaps_idx = np.where(diff_ttl > 5000)[0]

video_starts = np.append(video_ttls_tmp[0], video_ttls_tmp[gaps_idx+1])
video_stops = np.append(video_ttls_tmp[gaps_idx], video_ttls_tmp[-1])

video_ttls = {}
for i in range(len(video_starts)):
    idx = (video_ttls_tmp >= video_starts[i]) & (
        video_ttls_tmp <= video_stops[i])
    video_ttls[i] = video_ttls_tmp[idx]
    print(video_ttls[i].shape)
    # print(left_pupil_data[i]['com_x'].shape)

if experiment_name == '2022-12-20_15-49-59':
    # we need to skip the first video, it is short and probbably wrong
    video_ttls[0] = video_ttls[1]
    video_ttls[1] = video_ttls[2]
    video_ttls.pop(2)

plt.figure(8)
plt.clf()

ts = video_ttls_tmp
plt.plot(ts, np.ones(ts.shape)*i, '.')

ts = video_starts
plt.plot(ts, np.ones(ts.shape)*i, '.m')

ts = video_stops
plt.plot(ts, np.ones(ts.shape)*i, '.c')

for vid in video_ttls.keys():
    ts = video_ttls[vid]

    plt.plot(ts, np.ones(ts.shape)*3, '.')

ts = opto_left_starts
plt.plot(ts, np.ones(ts.shape)*4, '.')

ts = opto_right_starts
plt.plot(ts, np.ones(ts.shape)*5, '.')

ts = opto_left_starts[12]
plt.plot(ts, np.ones(ts.shape)*4, 'o')

# %%
# %%
right_pupil_data = data_pupil['right_pupil_data']
left_pupil_data = data_pupil['left_pupil_data']


thr = 0.9
com_x_re_ma = {}
com_x_le_ma = {}

com_y_re_ma = {}
com_y_le_ma = {}

x_rp_ma = {}
x_lp_ma = {}
x_nose_ma = {}

for vid in video_ttls.keys():
    # right eye
    com_x = right_pupil_data[vid]['com_x']
    com_y = right_pupil_data[vid]['com_y']
    com_p = right_pupil_data[vid]['com_p']

    mask = com_p <= thr
    com_x_re_ma[vid] = np.ma.masked_array(com_x, mask)
    com_y_re_ma[vid] = np.ma.masked_array(com_y, mask)

    # left eye
    com_x = left_pupil_data[vid]['com_x']
    com_y = left_pupil_data[vid]['com_y']
    com_p = left_pupil_data[vid]['com_p']

    mask = com_p <= thr
    com_x_le_ma[vid] = np.ma.masked_array(com_x, mask)
    com_y_le_ma[vid] = np.ma.masked_array(com_y, mask)

# %%

events_to_average = {'opto_right': opto_right_starts,
                     'opto_left': opto_left_starts}


com_x_events_re_all_events = {}
com_x_events_le_all_events = {}
com_y_events_re_all_events = {}
com_y_events_le_all_events = {}

pixy_angle_events = {}
pixy_movement_events = {}

opto_right_cont_events = {}
opto_left_cont_events = {}


x_lp_all_events = {}
x_rp_all_events = {}
x_nose_all_events = {}

for i, key in enumerate(events_to_average):

    #    ts = events[i]
    # ts = opto_right_starts
    # ts = opto_left_starts
    # ts = opto_left_starts
    ts = events_to_average[key]

    start_s = -2
    stop_s = 15
    frames = np.arange(start_s*150, stop_s*150, 1)
    n_frames = len(frames)
    com_x_events_re = np.ma.zeros([n_frames, len(ts)])
    com_x_events_le = np.ma.zeros([n_frames, len(ts)])
    com_y_events_re = np.ma.zeros([n_frames, len(ts)])
    com_y_events_le = np.ma.zeros([n_frames, len(ts)])

    x_events_rp = np.ma.zeros([n_frames, len(ts)])
    x_events_lp = np.ma.zeros([n_frames, len(ts)])
    x_events_nose = np.ma.zeros([n_frames, len(ts)])

    for j, ttl in enumerate(ts):
        print(ttl)

        # we have to go through the videos
        for vid in video_ttls.keys():
            video_ttls_tmp = video_ttls[vid]

            search_idx = np.searchsorted(video_ttls_tmp, ttl)

            if search_idx == 0:
                continue
            if search_idx == len(video_ttls_tmp):
                continue
            print(str(vid)+' - '+str(search_idx))

            # we cutout data
            frames_tmp = frames + search_idx

            com_x_events_re[:, j] = com_x_re_ma[vid][frames_tmp]
            com_x_events_le[:, j] = com_x_le_ma[vid][frames_tmp]
            com_y_events_re[:, j] = com_y_re_ma[vid][frames_tmp]
            com_y_events_le[:, j] = com_y_le_ma[vid][frames_tmp]

            # x_events_rp[:,j] = x_rp_ma[vid][frames_tmp]
            # x_events_lp[:,j] = x_lp_ma[vid][frames_tmp]
            # x_events_nose[:,j] = x_nose_ma[vid][frames_tmp]

    com_x_events_re_all_events[key] = com_x_events_re
    com_x_events_le_all_events[key] = com_x_events_le
    com_y_events_re_all_events[key] = com_y_events_re
    com_y_events_le_all_events[key] = com_y_events_le

    x_lp_all_events[key] = x_events_lp
    x_rp_all_events[key] = x_events_rp
    x_nose_all_events[key] = x_events_nose

    # # pixy stuff
    sr_raw = 30000
    sr_subsampled = np.int64(1/(np.mean(np.diff(timestamps_cont))/sr_raw))
    resolution_s = 0.001
    resolution = np.int64(resolution_s * sr_subsampled)
    data_idxs = np.arange(start_s*sr_subsampled,
                          stop_s*sr_subsampled, resolution)
    n_data = len(data_idxs)

    # pixy_angle_tmp = np.ma.zeros([n_data,len(ts)])
    # pixy_movement_tmp = np.ma.zeros([n_data,len(ts)])
    opto_right_cont_tmp = np.ma.zeros([n_data, len(ts)])
    opto_left_cont_tmp = np.ma.zeros([n_data, len(ts)])

    for j, ttl in enumerate(ts):
        # print(ttl)
        ttl_cont_index = np.searchsorted(timestamps_cont, ttl)

        # we cutout data
        data_idxs_tmp = data_idxs + ttl_cont_index
        # pixy_angle_tmp[:,j] = pixy_angle[data_idxs_tmp]
        # pixy_movement_tmp[:,j] = pixy_movement[data_idxs_tmp]

        opto_right_cont_tmp[:, j] = cont_opto_right[data_idxs_tmp]
        opto_left_cont_tmp[:, j] = cont_opto_left[data_idxs_tmp]

    # pixy_angle_events[key] = pixy_angle_tmp
    # pixy_movement_events[key] = pixy_movement_tmp

    opto_right_cont_events[key] = opto_right_cont_tmp
    opto_left_cont_events[key] = opto_left_cont_tmp


data_time_ms = data_idxs  # / 30.

# %%
video_sr = 1./150
frames_s = frames * video_sr
frames_ms = frames_s * 1000

idx_onset_values = (frames_ms > -20) & (frames_s <= 0)

# %
fig = plt.figure(524)
# fig.set_size_inches(7,10)
fig.set_size_inches(8, 7)
plt.clf()

key = 'opto_left'
# right eye, opto left
ax = plt.subplot(3, 2, 1)
tmp = com_x_events_re_all_events[key]
for j in range(tmp.shape[1]):
    y = tmp[:, j].copy()
    y -= y[idx_onset_values].mean()
    plt.plot(frames_ms, y+j*10)

# plt.plot(frames_ms,np.mean(tmp,1),'k',lw=2)
plt.title('Right Eye, Left Opto')
plt.plot([0, 0], plt.ylim(), 'k:')
plt.ylabel('Eye X position')


# left eye, opto left
plt.subplot(3, 2, 2, sharex=ax, sharey=ax)
tmp = com_x_events_le_all_events[key]
for j in range(tmp.shape[1]):
    y = tmp[:, j].copy()
    y -= y[idx_onset_values].mean()
    plt.plot(frames_ms, y+j*10)
# plt.plot(frames_ms,np.mean(tmp,1),'k',lw=2)
plt.title('Left Eye, Left Opto')
plt.plot([0, 0], plt.ylim(), 'k:')
plt.ylabel('Eye X position')


key = 'opto_right'
# right eye, opto right
plt.subplot(3, 2, 3, sharex=ax, sharey=ax)
tmp = com_x_events_re_all_events[key]
for j in range(tmp.shape[1]):
    y = tmp[:, j].copy()
    y -= y[idx_onset_values].mean()
    plt.plot(frames_ms, y+j*10)
# plt.plot(frames_ms,np.mean(tmp,1),'k',lw=2)
plt.title('Right Eye, Right Opto')
plt.plot([0, 0], plt.ylim(), 'k:')
plt.ylabel('Eye X position')


# left eye, opto right
plt.subplot(3, 2, 4, sharex=ax, sharey=ax)
tmp = com_x_events_le_all_events[key]
for j in range(tmp.shape[1]):
    y = tmp[:, j].copy()
    y -= y[idx_onset_values].mean()
    plt.plot(frames_ms, y+j*10)
plt.title('Left Eye, Right Opto')
plt.plot([0, 0], plt.ylim(), 'k:')

plt.ylabel('Eye X position')
plt.ylim([-50, tmp.shape[1]*12])


plt.subplot(3, 2, 5, sharex=ax)
key = 'opto_left'
# right eye, opto left
tmp = com_x_events_re_all_events[key]
tmp_mean = np.mean(tmp, 1)
tmp_mean -= tmp_mean[idx_onset_values].mean()
plt.plot(frames_ms, tmp_mean)

key = 'opto_right'
# right eye, opto right
tmp = com_x_events_re_all_events[key]
tmp_mean = np.mean(tmp, 1)
tmp_mean -= tmp_mean[idx_onset_values].mean()
plt.plot(frames_ms, tmp_mean)
plt.xlabel('Time (ms)')
plt.legend(['Opto left', 'Opto right'])
plt.title('Right Eye Mean')
plt.ylabel('Temporal - Nasal')
plt.plot([0, 0], plt.ylim(), 'k:')

plt.subplot(3, 2, 6, sharex=ax)
key = 'opto_left'
# left eye, opto left
tmp = com_x_events_le_all_events[key]
tmp_mean = np.mean(tmp, 1)
tmp_mean -= tmp_mean[idx_onset_values].mean()
plt.plot(frames_ms, tmp_mean)

key = 'opto_right'
# left eye, opto right
tmp = com_x_events_le_all_events[key]
tmp_mean = np.mean(tmp, 1)
tmp_mean -= tmp_mean[idx_onset_values].mean()
plt.plot(frames_ms, tmp_mean)

plt.legend(['Opto left', 'Opto right'])

plt.xlabel('Time (ms)')
plt.title('Left Eye Mean')
plt.ylabel('Nasal - Temporal')

plt.xlim([-1000, 5000])
plt.plot([0, 0], plt.ylim(), 'k:')

fig.tight_layout()
plt.draw()

plt.savefig(experiment_name+'_all_trial_left_right_opto.png', dpi=200)


# %%
data_time_ms = data_idxs  # < / 30.

video_sr = 1./150
frames_s = frames * video_sr
frames_ms = frames_s * 1000

idx_onset_values = (frames_ms > -20) & (frames_s <= 0)


key = 'opto_left'
# key = 'opto_right'

for key in ['opto_left', 'opto_right']:

    n_trials = opto_right_cont_events[key].shape[1]

    # trial = 27

    for trial in range(n_trials):
        idx_onset_values_frames = (frames_ms > -20) & (frames_s <= 0)
        idx_onset_values_cont = (data_time_ms > -20) & (data_time_ms <= 0)

        fig = plt.figure(212)
        fig.set_size_inches(5, 7)
        plt.clf()

        ax = plt.subplot(3, 1, 1)
        tmp = opto_right_cont_events[key][:, trial]
        plt.plot(data_time_ms, tmp, 'm')
        tmp = opto_left_cont_events[key][:, trial]
        plt.plot(data_time_ms, tmp, 'g')

        plt.legend(['opto right', 'opto left'])
        plt.title(experiment_name+', '+key + ' trial: '+str(trial))

        # tmp = com_x_events_re_all_events[key][:,trial]
        # tmp -= tmp[idx_onset_values_frames].mean()
        # plt.plot(frames_ms,tmp,'b')
        # plt.title(key + ' trial: '+str(trial))
        # plt.ylim([-30,30])
        # plt.plot([0,0],plt.ylim(),'k:')
        # plt.ylabel('Pupil x')

        plt.subplot(3, 1, 2, sharex=ax)
        tmp = com_x_events_le_all_events[key][:, trial]
        tmp -= tmp[idx_onset_values_frames].mean()
        plt.plot(frames_ms, tmp, 'r')
        tmp = com_x_events_re_all_events[key][:, trial]
        tmp -= tmp[idx_onset_values_frames].mean()
        plt.plot(frames_ms, tmp, 'b')
        plt.ylim([-50, 50])
        plt.plot([0, 0], plt.ylim(), 'k:')
        plt.ylabel('Pupil x')
        plt.legend(['LE', 'RE'])

        plt.subplot(3, 1, 3, sharex=ax)
        tmp = com_y_events_le_all_events[key][:, trial]
        tmp -= tmp[idx_onset_values_frames].mean()
        plt.plot(frames_ms, tmp, 'r')
        tmp = com_y_events_re_all_events[key][:, trial]
        tmp -= tmp[idx_onset_values_frames].mean()
        plt.plot(frames_ms, tmp, 'b')
        # plt.title(key + ' trial: '+str(trial))
        plt.ylim([-50, 50])
        plt.plot([0, 0], plt.ylim(), 'k:')
        plt.ylabel('Pupil y')
        plt.legend(['LE', 'RE'])

        # plt.subplot(5,1,4,sharex=ax)
        # tmp = pixy_movement_events[key][:,trial]
        # tmp -= tmp[idx_onset_values].mean()
        # plt.plot(data_time_ms,tmp,color=[0.5,0.5,0.5])
        # plt.ylim([-5000,5000])
        # plt.plot([0,0],plt.ylim(),'k:')
        # plt.ylabel('Pixy Movement')

        # plt.subplot(5,1,5,sharex=ax)
        # tmp = pixy_angle_events[key][:,trial]
        # tmp -= tmp[idx_onset_values].mean()
        # plt.plot(data_time_ms,tmp,color=[0.5,0.5,0.5])
        # plt.plot([0,0],plt.ylim(),'k:')
        # plt.ylabel('Pixy Angle')

        plt.xlabel('Time (ms)')

        plt.xlim([-1000, 5000])

        fig.tight_layout()
        plt.draw()

        plt.savefig('figures/'+experiment_name+'_' + key +
                    '_trial_' + str(trial).zfill(3) + '.png', dpi=200)
