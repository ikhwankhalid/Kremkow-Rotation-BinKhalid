#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 05:37:25 2022

@author: jenskremkow
"""

# %%
# %%
import pandas
import numpy as np
import matplotlib.pyplot as plt
import os
# import cv2 as cv2
import scipy.ndimage as ndimage
import numpy.ma as ma
from scipy import interpolate
import pycorrelate as pyc
from scipy.signal import butter, filtfilt
from scipy import interpolate
from scipy.interpolate import CubicSpline
from matplotlib import rcParams


def calc_euclidean_distance(rf_center_1, rf_center_2):
    rf_center_1 = np.array(rf_center_1)
    rf_center_2 = np.array(rf_center_2)

    euclidean_distance = np.linalg.norm(rf_center_1-rf_center_2)
    # distance can be negative, depending on the order of the centers. However, we are only interested in the absolute distance
    euclidean_distance = np.abs(euclidean_distance)

    return euclidean_distance


def calc_eye_mov(eye_data, thr=0.9):
    names = ['pupil1', 'pupil2', 'pupil3', 'pupil4',
             'pupil5', 'pupil6', 'pupil7', 'pupil8']  # ,'pupil7']

    eye_data_tmp = {}
    for name in names:
        tmp = np.array(eye_data['mouse1'][name])
        eye_data_tmp[name] = tmp

    names_pupil = ['pupil1', 'pupil2', 'pupil3',
                   'pupil4', 'pupil5', 'pupil6', 'pupil7', 'pupil8']

    n_frames = eye_data_tmp['pupil1'].shape[0]

    com_x = np.zeros([n_frames])
    com_y = np.zeros([n_frames])
    com_p = np.zeros([n_frames])

    pupil_size = np.zeros([n_frames])

    for i in range(n_frames):

        xs = []
        ys = []
        ps = []
        for name in names_pupil:
            x = eye_data_tmp[name][i, 0]
            y = eye_data_tmp[name][i, 1]
            p = eye_data_tmp[name][i, 2]

            if p > thr:
                xs.append(x)
                ys.append(y)
                ps.append(p)
        #
        if len(xs) <= 2:
            xs = [-10]  # com_x[-1]]
            ys = [-10]  # com_y[-1]]
            ps = [0]

        com_x[i] = np.mean(xs)
        com_y[i] = np.mean(ys)
        com_p[i] = np.mean(ps)

        # % size is the distance between the points and the center
        if len(xs) <= 2:
            pupil_size[i] = 0.
        else:
            ds = []
            center = [com_x[i], com_y[i]]
            for k in range(len(xs)):
                point = [xs[k], ys[k]]
                dtmp = np.abs(calc_euclidean_distance(center, point))
                ds.append(dtmp)
            pupil_size[i] = np.mean(ds)
    return com_x, com_y, com_p, pupil_size


# %%

mouse = 'mouse_3'
recording_session_date = '2022-04-06_16-04-19'
nidaq_node = 'NI-DAQmx-103.0'

# base_dir = '/Users/jenskremkow/Desktop/Airtrack_behavior/data/'
base_dir = 'D:/airtrack_analysis/data/'

data_dir = base_dir + mouse+'_all/'+recording_session_date + '/'
video_dates = pandas.read_excel(data_dir + 'videonames.xls')

figures_dir = data_dir+'figures/'
if not os.path.exists(figures_dir):
    os.mkdir(figures_dir)


psths_dir = data_dir + '/'
# if not os.path.exists(figures_dir):
#     os.mkdir(figures_dir)
# %%
# because the Recode Node changes sometimes I changed the name of the folder to "nidaq".
spikes = np.load(data_dir+'spiketimes_idx.npy',
                 encoding='latin1', allow_pickle=True).item()


tlls_aligned = np.load(
    data_dir+'TTL.npy', encoding='latin1', allow_pickle=True).item()
camera_NP = tlls_aligned['cameras']
sync_NP = tlls_aligned['sync']


# %%

# %% MUA
binwidth_s = 0.25
binwidth_idx = binwidth_s * 30000.
psth_range = np.arange(sync_NP[0], sync_NP[-1], binwidth_idx)
psth_range_s = psth_range[0:-1]/30000.


chs = np.arange(384)

psth = np.zeros([len(chs), len(psth_range)-1])
psth_norm = psth.copy()

for ch in chs:
    st = spikes[ch+1]
    hs = np.histogram(st, psth_range)

    psth[ch, :] = hs[0]
    # norm
    psth_norm[ch, :] = psth[ch, :] / psth[ch, :].max()


# %%

timestamps_NIDAQ = np.load(
    data_dir+'nidaq/experiment1/recording1/events/'+nidaq_node+'/TTL_1/timestamps.npy')
channel_states_NIDAQ = np.load(
    data_dir+'nidaq/experiment1/recording1/events/'+nidaq_node+'/TTL_1/channel_states.npy')

timestamps_NIDAQ_continues = np.load(
    data_dir+'nidaq/experiment1/recording1/continuous/'+nidaq_node+'/timestamps.npy')
start_time = timestamps_NIDAQ_continues[0]
stop_time = timestamps_NIDAQ_continues[-1]


raw_file = data_dir+'nidaq/experiment1/recording1/continuous/' + \
    nidaq_node+'/continuous.dat'
n_ch = 8
mm = np.memmap(raw_file, dtype=np.int16, mode='r')
mm = mm.reshape((n_ch, -1), order='F')


states = [1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8]

ttls = {}

for state in states:
    ttls[state] = timestamps_NIDAQ[channel_states_NIDAQ == state]

sync = ttls[1]
camera = ttls[2]
rLED = ttls[3]
lLED = ttls[4]
reward = ttls[5]
opto = ttls[6]
lAir = ttls[7]
rAir = ttls[8]

rLED_off = ttls[-3]
lLED_off = ttls[-4]


trial_onsets_l = []
trial_onsets_r = []
trial_onsets = []
trial_offsets = []

trial_offsets.append(reward[0])
if len(rLED_off) == 0:
    trial_onsets.append(lLED.min())
elif len(lLED_off) == 0:
    trial_onsets.append(rLED.min())
else:
    trial_onsets.append(np.min([rLED.min(), lLED.min()]))

for i, reward_tmp in enumerate(reward[0:-1]):

    trial_offsets.append(reward[i+1])
    # rLED
    bigger_r,  = np.where(rLED > reward_tmp)
    # lLED
    bigger_l,  = np.where(lLED > reward_tmp)

    if len(bigger_r) > 0:
        rLED_after_last_reward = rLED[bigger_r[0]]
        rdt = rLED_after_last_reward-reward_tmp
    else:
        rLED_after_last_reward = []
        rdt = np.inf
    if len(bigger_l) > 0:
        lLED_after_last_reward = lLED[bigger_l[0]]
        ldt = lLED_after_last_reward-reward_tmp
    else:
        lLED_after_last_reward = []
        ldt = np.inf

    if rdt < ldt:
        trial_onsets_r.append(rLED_after_last_reward)
        trial_onsets.append(rLED_after_last_reward)
    if ldt < rdt:
        trial_onsets_l.append(lLED_after_last_reward)
        trial_onsets.append(lLED_after_last_reward)

trial_onsets = np.array(trial_onsets)
trial_offsets = np.array(trial_offsets)


# %% fixing camera for 20220222
# # 20220222
# start_camera = 34221
# camera = camera[start_camera::]
# camera_NP = camera_NP[start_camera::]


# %%
plt.figure(1, figsize=(20, 10))
plt.clf()

plt.plot(camera, np.ones(camera.shape)*2, '|k', label='camera')
plt.plot(rLED, np.ones(rLED.shape)*3, '|b', label='rLED')
# plt.plot(rLED_off,np.ones(rLED_off.shape)*3,'|r',label='rLED')
plt.plot(lLED, np.ones(lLED.shape)*4, '|m', label='lLED')
plt.plot(reward, np.ones(reward.shape)*5, '|k', label='reward')
plt.plot(opto, np.ones(opto.shape)*6, '|y', label='opto')
plt.plot(lAir, np.ones(lAir.shape)*7, '|', label='lAir')
plt.plot(rAir, np.ones(rAir.shape)*8, '|', label='rAir')
# plt.legend()

plt.plot(trial_onsets, np.ones(trial_onsets.shape)*4.3, '|', label='onsets')
plt.plot(trial_offsets, np.ones(trial_offsets.shape)*4.7, '|r', label='onsets')

yticks = [1, 2, 3, 4, 4.3, 4.7, 5, 6, 7, 8]
yticks_label = ['sync', 'camera', 'rLED', 'lLED',
                'Onset', 'Offset', 'Reward', 'Opto', 'lAir', 'rAir']
plt.yticks(yticks, yticks_label, fontsize=6.)

plt.plot([start_time, start_time], [0, 9])
plt.plot([stop_time, stop_time], [0, 9])

plt.savefig(figures_dir+'TTL_overview.png', dpi=200)

# %%


video_data = {}

# we just look at the first video #videodate in enumerate(video_dates.video_date):
for i in range(len(video_dates)):
    videodate = video_dates.video_date[i]
    videodate = videodate.replace(' ', '')
    print(videodate)

    # video_data[i] = {}
    data_tmp = {}

    # whisker
    whisker_fn = 'cam0_'+videodate + \
        'DLC_resnet50_whisker_tracker_v2Dec2shuffle1_1030000.h5'
    # left eye
    left_eye_fn = 'cam1_'+videodate + \
        'DLC_dlcrnetms5_PupilBothEyesJSSetupNov4shuffle1_200000_el.h5'
    # right eye
    right_eye_fn = 'cam2_'+videodate + \
        'DLC_dlcrnetms5_PupilBothEyesJSSetupNov4shuffle1_200000_el.h5'
    # platform
    platform_fn = 'cam4_'+videodate+'DLC_resnet50_angle_trackerDec10shuffle1_1030000.h5'

    # %
    # load the video data
    # whisker
    fn = data_dir+whisker_fn
    with_whisker = 0
    if os.path.exists(fn):
        dlc_tmp = pandas.read_hdf(fn)
        dlc = dlc_tmp.keys()[0][0]
        whisker_data = eval('dlc_tmp.'+dlc)
        # video_data[i]['whisker'] = whisker_data
        with_whisker = 1

    # left eye
    fn = data_dir+left_eye_fn
    with_left_eye = 0
    if os.path.exists(fn):
        dlc_tmp = pandas.read_hdf(fn)
        dlc = dlc_tmp.keys()[0][0]
        left_eye_data = eval('dlc_tmp.'+dlc)
        # video_data[i]['left_eye'] = left_eye_data
        with_left_eye = 1

    # right eye
    fn = data_dir+right_eye_fn
    with_right_eye = 0
    if os.path.exists(fn):
        dlc_tmp = pandas.read_hdf(fn)
        dlc = dlc_tmp.keys()[0][0]
        right_eye_data = eval('dlc_tmp.'+dlc)
        # video_data[i]['right_eye'] = right_eye_data
        with_right_eye = 1

    # platform
    fn = data_dir+platform_fn
    with_platform = 0
    if os.path.exists(fn):
        dlc_tmp = pandas.read_hdf(fn)
        dlc = dlc_tmp.keys()[0][0]
        platform_data = eval('dlc_tmp.'+dlc)
        # video_data[i]['platform'] = platform_data
        with_platform = 1
    # print(whisker_data.shape)
    # print(left_eye_data.shape)
    # print(right_eye_data.shape)
    # print(platform_data.shape)

    # print('camera:' + str(ttls[2].shape))

    # % video analysis
    thr = 0.7

    if with_whisker:
        R_tip = np.array(whisker_data.R_tip)
        L_tip = np.array(whisker_data.L_tip)

        mask = R_tip[:, 2] < thr
        R_tip_x = ma.masked_array(R_tip[:, 0], mask=mask)
        R_tip_y = ma.masked_array(R_tip[:, 1], mask=mask)
        data_tmp['R_tip_x'] = R_tip_x
        data_tmp['R_tip_y'] = R_tip_y

        mask = L_tip[:, 2] < thr
        L_tip_x = ma.masked_array(L_tip[:, 0], mask=mask)
        L_tip_y = ma.masked_array(L_tip[:, 1], mask=mask)
        data_tmp['L_tip_x'] = L_tip_x
        data_tmp['L_tip_y'] = L_tip_y

        platform_R = np.array(whisker_data.platform_R)
        mask = platform_R[:, 2] < thr
        platform_R_x = ma.masked_array(platform_R[:, 0], mask=mask)
        platform_R_y = ma.masked_array(platform_R[:, 1], mask=mask)

        data_tmp['platform_R_x'] = platform_R_x
        data_tmp['platform_R_y'] = platform_R_y

        platform_L = np.array(whisker_data.platform_L)
        mask = platform_L[:, 2] < thr
        platform_L_x = ma.masked_array(platform_L[:, 0], mask=mask)
        platform_L_y = ma.masked_array(platform_L[:, 1], mask=mask)
        data_tmp['platform_L_x'] = platform_L_x
        data_tmp['platform_L_y'] = platform_L_y

    if with_left_eye:
        com_x, com_y, com_p, pupil_size = calc_eye_mov(left_eye_data, thr=thr)
        mask = com_p < thr
        com_x_masked = ma.masked_array(com_x, mask=mask)
        com_y_masked = ma.masked_array(com_y, mask=mask)
        data_tmp['com_x_le'] = com_x_masked
        data_tmp['com_y_le'] = com_y_masked

    if with_right_eye:
        com_x, com_y, com_p, pupil_size = calc_eye_mov(right_eye_data, thr=thr)
        mask = com_p < thr
        com_x_masked = ma.masked_array(com_x, mask=mask)
        com_y_masked = ma.masked_array(com_y, mask=mask)
        data_tmp['com_x_re'] = com_x_masked
        data_tmp['com_y_re'] = com_y_masked

    if with_platform:
        star = np.array(platform_data.star)
        circle = np.array(platform_data.circle)
        square = np.array(platform_data.square)
        triangle = np.array(platform_data.triangle)

        mask = star[:, 2] < thr
        star_x = ma.masked_array(star[:, 0], mask=mask)
        star_y = ma.masked_array(star[:, 1], mask=mask)
        data_tmp['star_x'] = star_x
        data_tmp['star_y'] = star_y

        mask = circle[:, 2] < thr
        circle_x = ma.masked_array(circle[:, 0], mask=mask)
        circle_y = ma.masked_array(circle[:, 1], mask=mask)
        data_tmp['circle_x'] = circle_x
        data_tmp['circle_y'] = circle_y

        mask = square[:, 2] < thr
        square_x = ma.masked_array(square[:, 0], mask=mask)
        square_y = ma.masked_array(square[:, 1], mask=mask)
        data_tmp['square_x'] = square_x
        data_tmp['square_y'] = square_y

        mask = triangle[:, 2] < thr
        triangle_x = ma.masked_array(triangle[:, 0], mask=mask)
        triangle_y = ma.masked_array(triangle[:, 1], mask=mask)
        data_tmp['triangle_x'] = triangle_x
        data_tmp['triangle_y'] = triangle_y

    video_data[i] = data_tmp
# %% concact the dlc data
com_x_le_all = np.array([])
com_y_le_all = np.array([])

com_x_re_all = np.array([])
com_y_re_all = np.array([])

R_tip_x_all = np.array([])
R_tip_y_all = np.array([])

platform_R_x_all = np.array([])
platform_R_y_all = np.array([])

n_dlc = 0
n_camera = len(camera)
for i in video_data:
    print(i)

    if with_whisker:
        R_tip_x_all = np.append(R_tip_x_all, video_data[i]['R_tip_x'])
        R_tip_y_all = np.append(R_tip_y_all, video_data[i]['R_tip_y'])

        platform_R_x_all = np.append(
            platform_R_x_all, video_data[i]['platform_R_x'])
        platform_R_y_all = np.append(
            platform_R_y_all, video_data[i]['platform_R_y'])

    if with_left_eye:
        com_x_le_all = np.append(com_x_le_all, video_data[i]['com_x_le'])
        com_y_le_all = np.append(com_y_le_all, video_data[i]['com_y_le'])
    if with_right_eye:
        com_x_re_all = np.append(com_x_re_all, video_data[i]['com_x_re'])
        com_y_re_all = np.append(com_y_re_all, video_data[i]['com_y_re'])

n_dlc = len(com_x_le_all)

np.save('com_x_le_all.npy', com_x_le_all, allow_pickle=True, fix_imports=True)

# %%

if n_dlc > n_camera:
    com_x_le_all = com_x_le_all[0:n_camera]
    com_y_le_all = com_y_le_all[0:n_camera]
    com_x_re_all = com_x_re_all[0:n_camera]
    com_y_re_all = com_y_re_all[0:n_camera]
    R_tip_x_all = R_tip_x_all[0:n_camera]
    R_tip_y_all = R_tip_y_all[0:n_camera]
    platform_R_x_all = platform_R_x_all[0:n_camera]
    platform_R_y_all = platform_R_y_all[0:n_camera]


# %%
eye_window = 40
x_me = [np.median(com_x_le_all), np.median(com_x_re_all)]
lim_x_eye = [np.min(x_me)-eye_window, np.max(x_me)+eye_window]

y_me = [np.median(com_y_le_all), np.median(com_y_re_all)]
lim_y_eye = [np.min(y_me)-eye_window, np.max(y_me)+eye_window]

left_eye_x_dict = {}
left_eye_x_dict['com_x_le_all'] = com_x_le_all
np.save(figures_dir+'com_x_le_all.npy', left_eye_x_dict)

left_eye_y_dict = {}
left_eye_y_dict['com_y_le_all'] = com_y_le_all
np.save(figures_dir+'com_y_le_all.npy', left_eye_y_dict)

right_eye_x_dict = {}
right_eye_x_dict['com_x_re_all'] = com_x_re_all
np.save(figures_dir+'com_x_re_all.npy', right_eye_x_dict)

right_eye_y_dict = {}
right_eye_y_dict['com_y_re_all'] = com_y_re_all
np.save(figures_dir+'com_y_re_all.npy', right_eye_y_dict)


# data_tmp['com_x'] = com_x
# data_tmp['com_y'] = com_y
# data_tmp['com_x_masked'] = com_x_masked
# data_tmp['com_y_masked'] = com_y_masked
# data_tmp['com_p'] = com_p
# data_tmp['pupil_size'] = pupil_size
#
# time_ms_eye = np.arange(com_x.shape[0])
# indx_good = com_p > 0.9
# cs = CubicSpline(time_ms_eye[indx_good], com_x[indx_good])
# com_x_interp = cs(time_ms_eye)
# cs = CubicSpline(time_ms_eye[indx_good], com_y[indx_good])
# com_y_interp = cs(time_ms_eye)
# data_tmp['com_x_interp'] = com_x_interp
# data_tmp['com_y_interp'] = com_y_interp

# left_eye_movements = data_tmp

# right eye movements
# com_x, com_y, com_p, pupil_size = calc_eye_mov(right_eye_data,thr=0.6)
# mask = com_p < thr
# com_x_masked = ma.masked_array(com_x, mask=mask)
# com_y_masked = ma.masked_array(com_y, mask=mask)
#
# data_tmp = {}
# data_tmp['com_x'] = com_x
# data_tmp['com_y'] = com_y
# data_tmp['com_x_masked'] = com_x_masked
# data_tmp['com_y_masked'] = com_y_masked
# data_tmp['com_p'] = com_p
# data_tmp['pupil_size'] = pupil_size
#
# time_ms_eye = np.arange(com_x.shape[0])
# indx_good = com_p > 0.9
# cs = CubicSpline(time_ms_eye[indx_good], com_x[indx_good])
# com_x_interp = cs(time_ms_eye)
# cs = CubicSpline(time_ms_eye[indx_good], com_y[indx_good])
# com_y_interp = cs(time_ms_eye)
# data_tmp['com_x_interp'] = com_x_interp
# data_tmp['com_y_interp'] = com_y_interp

# right_eye_movements = data_tmp


# %%

with_DLC = 1
with_right_eye = 1
with_left_eye = 1

rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 6
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

rcParams['axes.linewidth'] = 0.5
rcParams['lines.linewidth'] = 0.5

rcParams['xtick.major.width'] = 0.5
rcParams['ytick.major.width'] = 0.5


to_inch = 0.393701
fwidth_cm = 25.  # cm
fheight_cm = 40.  # cm
tick_length = 3.
lw = 0.5

fwidth_inch = fwidth_cm * to_inch
fheight_inch = fheight_cm * to_inch

color_x = 'k'
color_y = 'm'

color_left = 'r'
color_right = 'b'


window_before_after = 20000
window_before_after = 20000

start = timestamps_NIDAQ_continues[0]
stop = timestamps_NIDAQ_continues[-1]

window_s = 5 * 60
window_idx = window_s * 30000
chunks = np.arange(start, stop+window_idx, window_idx)


for i in range(len(chunks)-1):

    # trial_onset = trial_onsets[i]
    # trial_offset = trial_offsets[i]
    # time_start = trial_onset - window_before_after
    # time_stop = trial_offset + window_before_after
    # idx_start = trial_onset - start_time - window_before_after
    # idx_stop = trial_offset - start_time + window_before_after

    time_start = chunks[i]
    time_stop = chunks[i+1]

    # %

    idx_start = time_start - start_time
    idx_stop = time_stop - start_time

    xlim = [time_start, time_stop]

    tmp = mm[3, idx_start:idx_stop].copy()  # movement
    movement = ndimage.gaussian_filter1d(tmp, 120.)
    tmp = mm[4, idx_start:idx_stop].copy()  # angle
    angle = ndimage.gaussian_filter1d(tmp, 20.)

    time = timestamps_NIDAQ_continues[idx_start:idx_stop]

    fig = plt.figure(45)
    plt.clf()
    fig.set_size_inches(fwidth_inch, fheight_inch)

    # TTLs
    ax = plt.subplot(8, 1, 1)
    # for j in range(9):
    #    plt.plot([time[0],time[-1]],[j+1,j+1],'-',color='gray',lw=0.5)
    plt.plot(camera, np.ones(camera.shape)*2, '|k', label='camera')
    plt.plot(rLED, np.ones(rLED.shape)*3, '|b', label='rLED')
    plt.plot(rLED_off, np.ones(rLED_off.shape)*3, '|k', label='rLED')

    plt.plot(lLED, np.ones(lLED.shape)*4, '|m', label='lLED')
    plt.plot(lLED_off, np.ones(lLED_off.shape)*4, '|m', label='lLED')

    plt.plot(reward, np.ones(reward.shape)*5, '|m', label='reward')
    plt.plot(opto, np.ones(opto.shape)*6, '|y', label='opto')
    plt.plot(lAir, np.ones(lAir.shape)*7, '|', label='lAir')
    plt.plot(rAir, np.ones(rAir.shape)*8, '|', label='rAir')
    plt.plot(trial_onsets, np.ones(trial_onsets.shape)*9, '|g', label='onsets')
    plt.plot(trial_offsets, np.ones(trial_offsets.shape)
             * 10, '|r', label='offsets')

    # plt.legend()
    yticks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    yticks_label = ['Sync', 'camera', 'rLED', 'lLED',
                    'Reward', 'Opto', 'lAir', 'rAir', 'Onset', 'Offset']
    plt.yticks(yticks, yticks_label, fontsize=6.)

    plt.ylim([0, 11])
    # plt.xticks(xticks,[])

    plt.xlim(xlim)

    n_camera_frames = len(camera)
    # left eye
    # com_x = left_eye_movements['com_x_masked'][0:n_camera_frames]
    # com_y = left_eye_movements['com_y_masked'][0:n_camera_frames]

    ax = plt.subplot(8, 1, 2, sharex=ax)
    if with_left_eye:
        plt.plot(camera, com_x_le_all, color=color_left)
        plt.plot(camera, com_x_re_all, color=color_right)

    # plt.ylim(lim_x_eye)
    plt.ylabel('Horizontal eye movements')

    # right eye
    # com_x = right_eye_movements['com_x_masked'][0:n_camera_frames]
    # com_y = right_eye_movements['com_y_masked'][0:n_camera_frames]

    ax = plt.subplot(8, 1, 3, sharex=ax)
    if with_right_eye:
        plt.plot(camera, com_y_le_all, color=color_left)
        plt.plot(camera, com_y_re_all, color=color_right)

    # plt.ylim(lim_y_eye)
    plt.ylabel('Vertical eye movements')

    # whisker R
    ax = plt.subplot(8, 1, 7, sharex=ax)
    # x_tmp = R_tip[0:n_camera_frames,0]
    # y_tmp = R_tip[0:n_camera_frames,1]
    if with_whisker:
        # x_tmp = R_tip_x[0:n_camera_frames]
        # y_tmp = R_tip_y[0:n_camera_frames]

        plt.plot(camera, R_tip_x_all, color=color_x)
        # plt.plot(camera,y_tmp,color=color_y)

    plt.ylabel('Whisker tip right (dlc)')

    # plotform right
    ax = plt.subplot(8, 1, 8, sharex=ax)
    # x_tmp = platform_R[0:n_camera_frames,0]
    # y_tmp = platform_R[0:n_camera_frames,1]
    if with_whisker:
        # x_tmp = platform_R_x[0:n_camera_frames]
        # y_tmp = platform_R_y[0:n_camera_frames]

        plt.plot(camera, platform_R_x_all, color=color_x)
        # plt.plot(camera,y_tmp,color=color_y)

    plt.ylabel('Platform position right (dlc)')

    ax = plt.subplot(8, 1, 5, sharex=ax)
    plt.plot(time[0::100], movement[0::100], 'k', lw=lw)
    plt.ylabel('Platform movement (pixy)')

    ax = plt.subplot(8, 1, 6, sharex=ax)
    plt.plot(time[0::100], angle[0::100], 'k', lw=lw)
    plt.ylabel('Platform angle (pixy)')

    # MUA
    ax = plt.subplot(8, 1, 4, sharex=ax)
    plt.pcolormesh(psth_range[0:-1], chs, psth_norm,
                   rasterized=True, cmap=plt.cm.magma)
    plt.ylabel('MUA channel')

    plt.xlim(xlim)

    fig.tight_layout()

    figure_name = figures_dir+recording_session_date+'_slice_'+str(i).zfill(3)
    plt.savefig(figure_name+'.png', dpi=300)
# %% eye movements


# %% movement analysis

tmp = mm[3, :].copy()  # movement
movement = ndimage.gaussian_filter1d(tmp, 120.)

# %% platform movement (pixy) and PSTH

thr_value = 6000
thr_idx, = np.where(movement == thr_value)

gaps, = np.where(np.diff(thr_idx) > 30000)  # 30 0000 IS actually 1 second
thr_idx_new = thr_idx[gaps+1]

# %
thr_times = timestamps_NIDAQ_continues[thr_idx_new]

d_mov = movement[thr_idx_new+30000]-movement[thr_idx_new]

up_movement_idx = thr_idx_new[d_mov > 0]
down_movement_idx = thr_idx_new[d_mov < 0]

up_movement_time = thr_times[d_mov > 0]
down_movement_time = thr_times[d_mov < 0]

# %
sampling_rate = 30000

window_mta = np.int(5.*sampling_rate)

window_index = np.arange(-window_mta, window_mta, 1)
window_s = window_index/sampling_rate

up_value = np.zeros([len(window_index), len(up_movement_idx)])
up_time = np.zeros([len(window_index), len(up_movement_idx)])

down_value = np.zeros([len(window_index), len(down_movement_idx)])
down_time = np.zeros([len(window_index), len(down_movement_idx)])

up_mov = np.zeros([len(window_index), len(up_movement_idx)])
up_mov_time = np.zeros([len(window_index), len(up_movement_idx)])
down_mov = np.zeros([len(window_index), len(down_movement_idx)])
down_mov_time = np.zeros([len(window_index), len(down_movement_idx)])

up_movement_time_for_MTA_spikes = []
down_movement_time_for_MTA_spikes = []

count_up = 0
for i, index_stamp in enumerate(up_movement_idx):
    # idx = np.searchsorted(timestamps_NP_s, time_stamp)
    idxs = index_stamp+window_index
    # up_mov_time[:,i] = timestamps_NP_s[idxs]
    if idxs.min() <= 0:
        continue
    if idxs.max() > len(movement):
        continue
    up_mov[:, count_up] = movement[idxs]

    up_movement_time_for_MTA_spikes.append(up_movement_time[i])
    count_up += 1

count_down = 0
for i, index_stamp in enumerate(down_movement_idx):
    # idx = np.searchsorted(timestamps_NP_s, time_stamp)
    idxs = index_stamp+window_index
    # up_mov_time[:,i] = timestamps_NP_s[idxs]
    if idxs.min() <= 0:
        continue
    if idxs.max() > len(movement):
        continue
    down_mov[:, count_down] = movement[idxs]

    down_movement_time_for_MTA_spikes.append(down_movement_time[i])
    count_down += 1

count_min = np.min([count_up, count_down])
up_mov = up_mov[:, 0:count_min]
down_mov = down_mov[:, 0:count_min]

up_movement_idx = up_movement_idx[0:count_min]
down_movement_idx = down_movement_idx[0:count_min]

# % movemtn triggered average
psth_range_mov_s = np.arange(-5., 5, 0.1)
psth_range_mov_idx = psth_range_mov_s * 30000

psth_mov_up = np.zeros([len(chs), len(psth_range_mov_idx)-1])
psth_mov_down = np.zeros([len(chs), len(psth_range_mov_idx)-1])

psth_mov_up_norm = psth_mov_up.copy()
psth_mov_down_norm = psth_mov_down.copy()


for ch in chs:
    st = spikes[ch+1]

    # move up
    trial_count = 0
    for trial, timestamp in enumerate(up_movement_time_for_MTA_spikes):
        st_tmp = st - timestamp

        hs = np.histogram(st_tmp, psth_range_mov_idx)
        psth_mov_up[ch, :] = psth_mov_up[ch, :] + hs[0]
        trial_count += 1
    psth_mov_up[ch, :] /= np.float(trial_count)

    tmp = psth_mov_up[ch, :].copy()
    tmp -= tmp.min()
    tmp /= tmp.max()
    # psth_mov_up[ch,:] / psth_mov_up[ch,:].max()
    psth_mov_up_norm[ch, :] = tmp

    # move down
    trial_count = 0
    for trial, timestamp in enumerate(down_movement_time_for_MTA_spikes):
        st_tmp = st - timestamp

        hs = np.histogram(st_tmp, psth_range_mov_idx)
        psth_mov_down[ch, :] = psth_mov_down[ch, :] + hs[0]
        trial_count += 1
    psth_mov_down[ch, :] /= np.float(trial_count)

    tmp = psth_mov_down[ch, :].copy()
    tmp -= tmp.min()
    tmp /= tmp.max()
    psth_mov_down_norm[ch, :] = tmp

    # psth_mov_down_norm[ch,:] = psth_mov_down[ch,:] / psth_mov_down[ch,:].max()


# % angle analysis


tmp = mm[4, :].copy()  # angle
angle = ndimage.gaussian_filter1d(tmp, 20.)

# %

thr_value = 8000
thr_idx, = np.where(angle == thr_value)

gaps, = np.where(np.diff(thr_idx) > 30000)
thr_idx_new = thr_idx[gaps+1]

# %
thr_times = timestamps_NIDAQ_continues[thr_idx_new]

d_mov = angle[thr_idx_new+30000]-angle[thr_idx_new]

left_movement_idx = thr_idx_new[d_mov > 0]
right_movement_idx = thr_idx_new[d_mov < 0]

left_movement_time = thr_times[d_mov > 0]
right_movement_time = thr_times[d_mov < 0]


left_value = np.zeros([len(window_index), len(left_movement_idx)])
left_time = np.zeros([len(window_index), len(left_movement_idx)])

right_value = np.zeros([len(window_index), len(right_movement_idx)])
right_time = np.zeros([len(window_index), len(right_movement_idx)])

left_mov = np.zeros([len(window_index), len(left_movement_idx)])
left_mov_time = np.zeros([len(window_index), len(left_movement_idx)])
right_mov = np.zeros([len(window_index), len(right_movement_idx)])
right_mov_time = np.zeros([len(window_index), len(right_movement_idx)])

left_movement_time_for_MTA_spikes = []
right_movement_time_for_MTA_spikes = []

count_left = 0
for i, index_stamp in enumerate(left_movement_idx):
    # idx = np.searchsorted(timestamps_NP_s, time_stamp)
    idxs = index_stamp+window_index
    # up_mov_time[:,i] = timestamps_NP_s[idxs]
    if idxs.min() <= 0:
        continue
    if idxs.max() > len(angle):
        continue
    left_mov[:, count_left] = angle[idxs]

    left_movement_time_for_MTA_spikes.append(left_movement_time[i])
    count_left += 1

count_right = 0
for i, index_stamp in enumerate(right_movement_idx):
    # idx = np.searchsorted(timestamps_NP_s, time_stamp)
    idxs = index_stamp+window_index
    # up_mov_time[:,i] = timestamps_NP_s[idxs]
    if idxs.min() <= 0:
        continue
    if idxs.max() > len(angle):
        continue
    right_mov[:, count_right] = angle[idxs]

    right_movement_time_for_MTA_spikes.append(right_movement_time[i])
    count_right += 1

count_min = np.min([count_left, count_right])
left_mov = left_mov[:, 0:count_min]
right_mov = right_mov[:, 0:count_min]

left_movement_idx = left_movement_idx[0:count_min]
right_movement_idx = right_movement_idx[0:count_min]


# % angle triggered average
psth_range_mov_s = np.arange(-5., 5, 0.1)
psth_range_mov_idx = psth_range_mov_s * 30000

psth_mov_left = np.zeros([len(chs), len(psth_range_mov_idx)-1])
psth_mov_right = np.zeros([len(chs), len(psth_range_mov_idx)-1])

psth_mov_left_norm = psth_mov_up.copy()
psth_mov_right_norm = psth_mov_down.copy()


for ch in chs:
    st = spikes[ch+1]

    # move left
    trial_count = 0
    for trial, timestamp in enumerate(left_movement_time_for_MTA_spikes):
        st_tmp = st - timestamp

        hs = np.histogram(st_tmp, psth_range_mov_idx)
        psth_mov_left[ch, :] = psth_mov_left[ch, :] + hs[0]
        trial_count += 1
    psth_mov_left[ch, :] /= np.float(trial_count)

    tmp = psth_mov_left[ch, :].copy()
    tmp -= tmp.min()
    tmp /= tmp.max()
    # psth_mov_up[ch,:] / psth_mov_up[ch,:].max()
    psth_mov_left_norm[ch, :] = tmp

    # move right
    trial_count = 0
    for trial, timestamp in enumerate(right_movement_time_for_MTA_spikes):
        st_tmp = st - timestamp

        hs = np.histogram(st_tmp, psth_range_mov_idx)
        psth_mov_right[ch, :] = psth_mov_right[ch, :] + hs[0]
        trial_count += 1
    psth_mov_right[ch, :] /= np.float(trial_count)

    tmp = psth_mov_right[ch, :].copy()
    tmp -= tmp.min()
    tmp /= tmp.max()
    psth_mov_right_norm[ch, :] = tmp


# % reward
# sync = ttls[1]
# camera = ttls[2]
# rLED = ttls[3]
# lLED = ttls[4]
# reward = ttls[5]
# opto = ttls[6]
# lAir = ttls[7]
# rAir = ttls[8]
psth_range_reward_s = np.arange(-2., 2, 0.01)
psth_range_reward_idx = psth_range_reward_s * 30000

psth_reward = np.zeros([len(chs), len(psth_range_reward_idx)-1])
psth_reward_norm = psth_reward.copy()


psth_rLED = np.zeros([len(chs), len(psth_range_reward_idx)-1])
psth_rLED_norm = psth_rLED.copy()


psth_lLED = np.zeros([len(chs), len(psth_range_reward_idx)-1])
psth_lLED_norm = psth_lLED.copy()

psth_rAir = np.zeros([len(chs), len(psth_range_reward_idx)-1])
psth_rAir_norm = psth_rAir.copy()


psth_lAir = np.zeros([len(chs), len(psth_range_reward_idx)-1])
psth_lAir_norm = psth_lAir.copy()


for ch in chs:
    st = spikes[ch+1]

    # reward
    trial_count = 0
    for trial, timestamp in enumerate(reward):
        st_tmp = st - timestamp

        hs = np.histogram(st_tmp, psth_range_reward_idx)
        psth_reward[ch, :] = psth_reward[ch, :] + hs[0]
        trial_count += 1
    psth_reward[ch, :] /= np.float(trial_count)

    tmp = psth_reward[ch, :].copy()
    tmp -= tmp.min()
    tmp /= tmp.max()
    psth_reward_norm[ch, :] = tmp

    # rLED
    trial_count = 0
    for trial, timestamp in enumerate(rLED):
        st_tmp = st - timestamp

        hs = np.histogram(st_tmp, psth_range_reward_idx)
        psth_rLED[ch, :] = psth_rLED[ch, :] + hs[0]
        trial_count += 1
    psth_rLED[ch, :] /= np.float(trial_count)

    tmp = psth_rLED[ch, :].copy()
    tmp -= tmp.min()
    tmp /= tmp.max()
    psth_rLED_norm[ch, :] = tmp

    # lLED
    trial_count = 0
    for trial, timestamp in enumerate(lLED):
        st_tmp = st - timestamp

        hs = np.histogram(st_tmp, psth_range_reward_idx)
        psth_lLED[ch, :] = psth_lLED[ch, :] + hs[0]
        trial_count += 1
    psth_lLED[ch, :] /= np.float(trial_count)

    tmp = psth_lLED[ch, :].copy()
    tmp -= tmp.min()
    tmp /= tmp.max()
    psth_lLED_norm[ch, :] = tmp

    # rAir
    trial_count = 0
    for trial, timestamp in enumerate(rAir):
        st_tmp = st - timestamp

        hs = np.histogram(st_tmp, psth_range_reward_idx)
        psth_rAir[ch, :] = psth_rAir[ch, :] + hs[0]
        trial_count += 1
    psth_rAir[ch, :] /= np.float(trial_count)

    tmp = psth_rAir[ch, :].copy()
    tmp -= tmp.min()
    tmp /= tmp.max()
    psth_rAir_norm[ch, :] = tmp

    # lAir
    trial_count = 0
    for trial, timestamp in enumerate(lAir):
        st_tmp = st - timestamp

        hs = np.histogram(st_tmp, psth_range_reward_idx)
        psth_lAir[ch, :] = psth_lAir[ch, :] + hs[0]
        trial_count += 1
    psth_lAir[ch, :] /= np.float(trial_count)

    tmp = psth_lAir[ch, :].copy()
    tmp -= tmp.min()
    tmp /= tmp.max()
    psth_lAir_norm[ch, :] = tmp


# %
fig = plt.figure(46)
plt.clf()
fig.set_size_inches(10, 5.)

# movement
plt.subplot(2, 4, 1)
plt.plot(window_s, up_mov.mean(1))
plt.title('Platform movement backward')
plt.xlim([-4, 4])
plt.xlabel('Time (s)')


plt.subplot(2, 4, 5)
plt.pcolormesh(psth_range_mov_s, chs, psth_mov_up_norm, cmap=plt.cm.magma)
plt.xlim([-4, 4])
plt.xlabel('Time (s)')

plt.subplot(2, 4, 2)
plt.plot(window_s, down_mov.mean(1))
plt.title('Platform movement forward')
plt.xlim([-4, 4])
plt.xlabel('Time (s)')

plt.subplot(2, 4, 6)
plt.pcolormesh(psth_range_mov_s, chs, psth_mov_down_norm, cmap=plt.cm.magma)
plt.xlim([-4, 4])
plt.xlabel('Time (s)')

# angle
plt.subplot(2, 4, 3)
plt.plot(window_s, left_mov.mean(1))
plt.title('Platform angle left')
plt.xlim([-4, 4])
plt.xlabel('Time (s)')

plt.subplot(2, 4, 7)
plt.pcolormesh(psth_range_mov_s, chs, psth_mov_left_norm, cmap=plt.cm.magma)
plt.xlim([-4, 4])
plt.xlabel('Time (s)')

plt.subplot(2, 4, 4)
plt.plot(window_s, right_mov.mean(1))
plt.title('Platform angle right')
plt.xlim([-4, 4])
plt.xlabel('Time (s)')

plt.subplot(2, 4, 8)
plt.pcolormesh(psth_range_mov_s, chs, psth_mov_right_norm, cmap=plt.cm.magma)
plt.xlim([-4, 4])
plt.xlabel('Time (s)')

fig.tight_layout()


fname = data_dir + mouse + recording_session_date
plt.savefig(fname+'_platform_movement_angle.png', dpi=300)

# %

fig = plt.figure(47)
plt.clf()
fig.set_size_inches(10, 5.)

# reward
plt.subplot(1, 5, 1)
plt.pcolormesh(psth_range_reward_s, chs, psth_reward_norm, cmap=plt.cm.magma)
plt.xlim([-1.5, 1.5])
plt.title('Reward')
plt.xlabel('Time (s)')

# left LED
plt.subplot(1, 5, 2)
plt.pcolormesh(psth_range_reward_s, chs, psth_lLED_norm, cmap=plt.cm.magma)
plt.xlim([-0.2, 0.4])
plt.title('Left LED')
plt.xlabel('Time (s)')

# right LED
plt.subplot(1, 5, 3)
plt.pcolormesh(psth_range_reward_s, chs, psth_rLED_norm, cmap=plt.cm.magma)
plt.xlim([-0.2, 0.4])
plt.title('Right LED')
plt.xlabel('Time (s)')

# left Air
plt.subplot(1, 5, 4)
plt.pcolormesh(psth_range_reward_s, chs, psth_lAir_norm, cmap=plt.cm.magma)
plt.xlim([-0.2, 0.2])
plt.title('Left Air')
plt.xlabel('Time (s)')

# right Air
plt.subplot(1, 5, 5)
plt.pcolormesh(psth_range_reward_s, chs, psth_rAir_norm, cmap=plt.cm.magma)
plt.xlim([-0.2, 0.2])
plt.title('Right Air')
plt.xlabel('Time (s)')

fig.tight_layout()
fname = data_dir + mouse + recording_session_date
plt.savefig(fname+'_reward_LED_Air.png', dpi=300)


# %% Save your psths

# fnames = ['up', 'down', 'left', '', '']

fname_up = 'psth_mov_up.npy'
fname_down = 'psth_mov_down.npy'
fname_left = 'psth_mov_left.npy'
fname_right = 'psth_mov_right.npy'
fname_reward = 'psth_reward.npy'
fname_rLED = 'psth_rLED.npy'
fname_lLED = 'psth_lLED.npy'
fname_rAir = 'psth_rAir.npy'
fname_lAir = 'psth_lAir.npy'

np.save(psths_dir + fname_up, psth_mov_up)
np.save(psths_dir + fname_down, psth_mov_down)
np.save(psths_dir + fname_left, psth_mov_left)
np.save(psths_dir + fname_right, psth_mov_right)
np.save(psths_dir + fname_reward, psth_reward)
np.save(psths_dir + fname_rLED, psth_rLED)
np.save(psths_dir + fname_lLED, psth_lLED)
np.save(psths_dir + fname_rAir, psth_rAir)
np.save(psths_dir + fname_lAir, psth_lAir)

print('psths saved in ' + str(psths_dir))
