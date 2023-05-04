#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 22:02:54 2022

@author: kailun
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import os, re, cv2
plt.rcParams['animation.ffmpeg_path'] = 'C:/Users/AGKremkow/anaconda3/envs/DEEPLABCUT/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win64-v4.2.2.exe'

# helper functions

def getDataAndPreviewTtls(ttlKeys, cameraCh, refCh, refStartTtlIdx, nTtl, 
                          videoBreakThreshSec, ttlshifts=None, useSec=False, 
                          npixSamplingRate=30000):
    videoBreakThresh = videoBreakThreshSec * npixSamplingRate
    chStatePath = os.path.join(ttlDir, "channel_states.npy")
    timestampsPath = os.path.join(ttlDir, "timestamps.npy")
    chStates = np.load(chStatePath)
    timestamps = np.load(timestampsPath)
    ttls, uniquePveChs = getAllTimestamps(timestamps, chStates)
    camTtls, cameraChIdx = previewTimestamps(ttls, uniquePveChs, ttlKeys, cameraCh, useSec)
    (nVideos, videoBreaks, videoStarts, videoEnds, videoDurationsSec, 
     videoFrameRate) = getVideoInfo(camTtls, videoBreakThresh, npixSamplingRate)
    print(f"There are {nVideos} videos. The breaks are at {videoBreaks}.")
    print(f"The video durations (sec) are: {videoDurationsSec.round(2)}")
    refChIdx = np.where(uniquePveChs==refCh)[0][0]
    refTtls = ttls[refChIdx].copy()
    refStart = refTtls[refStartTtlIdx]
    refStopTtlIdx = refStartTtlIdx + nTtl
    if nTtl > len(refTtls):
        #refStart = int(round(refStartTtlIdx * videoFrameRate)) + videoStarts[0]
        #refStop = int(round(refStopTtlIdx * videoFrameRate)) + videoStarts[0]
        refStop = int(round(refStopTtlIdx * videoFrameRate)) + refStart
    else:
        refStop = refTtls[refStopTtlIdx]
    if ttlShifts is not None:
        refStart += int(round(ttlShifts * videoFrameRate))
        refStop += int(round(ttlShifts * videoFrameRate))
    previewVideoStartStop(videoStarts, videoEnds, refStart, refStop, cameraChIdx, refChIdx)
    isValidRef, videoIdx = hasValidStartStop(refStart, refStop, videoStarts, videoEnds)
    assert isValidRef, "The refStartTtlIdx and refStopTtlIdx are not valid!"
    vidInterestStart = videoStarts[videoIdx]
    vidInterestStop = videoEnds[videoIdx]
    vidInterestMask = (camTtls >= vidInterestStart) & (camTtls <= vidInterestStop)
    vidInterestTtls = camTtls[vidInterestMask]
    videoCropMask = (vidInterestTtls >= refStart) & (vidInterestTtls < refStop)
    return videoIdx, videoCropMask, videoFrameRate, ttls, uniquePveChs, camTtls, videoStarts, videoEnds, refStart, refStop

def getVideoInfo(ttls, videoBreakThresh, samplingRate):
    ttlDiffs = np.diff(ttls)
    videoBreakInds, = np.where(ttlDiffs >= videoBreakThresh)
    nVideos = len(videoBreakInds) + 1
    videoBreaks = ttls[videoBreakInds]
    videoStartInds = np.append(0, videoBreakInds+1)
    videoStarts = ttls[videoStartInds]
    videoEndInds = np.append(videoBreakInds, len(ttls)-1)
    videoEnds = ttls[videoEndInds]
    videoDurationsSec = (videoEnds - videoStarts) / samplingRate
    zeroDurationIdx, = np.where(videoDurationsSec == 0)
    nZero = len(zeroDurationIdx)
    if nZero > 0:
        nVideos -= nZero
        clippedInds = np.clip(zeroDurationIdx, 0, len(videoBreaks)-1)
        videoBreakInds = np.delete(videoBreakInds, clippedInds)
        videoBreaks = np.delete(videoBreaks, clippedInds)
        videoStartInds = np.delete(videoStartInds, zeroDurationIdx)
        videoEndInds = np.delete(videoEndInds, zeroDurationIdx)
        videoStarts = np.delete(videoStarts, zeroDurationIdx)
        videoEnds = np.delete(videoEnds, zeroDurationIdx)
        videoDurationsSec = np.delete(videoDurationsSec, zeroDurationIdx)
    nFrames = videoEndInds - videoStartInds + 1
    videoFrameRate = (nFrames / videoDurationsSec).mean()
    return nVideos, videoBreaks, videoStarts, videoEnds, videoDurationsSec, videoFrameRate

def getAllTimestamps(timestamps, chStates):
    uniqueChs = np.unique(chStates)
    positiveMask = uniqueChs > 0
    uniquePveChs = uniqueChs[positiveMask]
    ttls = []
    for c, targetCh in enumerate(uniquePveChs):
        targetMask = chStates == targetCh
        targetTimestamps = timestamps[targetMask]
        ttls.append(targetTimestamps)
    return ttls, uniquePveChs

def previewTimestamps(ttls, uniquePveChs, ttlKeys, cameraCh=2, useSec=False, 
                      npixSamplingRate=30000):
    nCh = len(ttls)
    startTime = min([min(t) for t in ttls])
    endTime = max([max(t) for t in ttls])
    times = np.linspace(startTime, endTime, 5)
    timesSec = (times-startTime) / npixSamplingRate
    cameraChIdx = np.where(uniquePveChs==cameraCh)[0][0]
    cameraTtls = ttls[cameraChIdx].copy()
    ttls[cameraChIdx] = []
    cmap = mpl.cm.rainbow
    colors = [cmap(i/nCh) for i in range(nCh)]
    plt.eventplot(ttls, colors=colors)
    plt.scatter(cameraTtls, np.zeros(len(cameraTtls))+cameraChIdx, color=colors[cameraChIdx], s=0.1)
    plt.yticks(np.arange(nCh), ttlKeys)
    if useSec:
        plt.xticks(times, timesSec.round(3))
        xlabel = "Time (s)"
    else:
        xlabel = "Times (unit time)"
    plt.xlabel(xlabel)
    return cameraTtls, cameraChIdx

def previewVideoStartStop(videoStarts, videoStops, refStart, refStop, cameraChIdx, refChIdx):
    size = 10
    camYs = np.ones(len(videoStarts)) * cameraChIdx + 0.2
    refYs = np.ones(2) * refChIdx
    plt.scatter(videoStarts, camYs, c='r', s=size)
    plt.scatter(videoStops, camYs, c='k', s=size)
    plt.scatter([refStart, refStop], refYs, c='g', s=size, zorder=100)

def hasValidStartStop(start, stop, videoStarts, videoStops):
    assert start < stop, f"The start ({start}) is larger than stop ({stop})!"
    validStart = False
    validStop = False
    isValid = False
    videoIdx = None
    for s, vidStart in enumerate(videoStarts):
        vidStop = videoStops[s]
        if (start >= vidStart) & (start <= vidStop):
            validStart = True
        if (stop >= vidStart) & (stop <= vidStop):
            validStop = True
        if validStart & validStop:
            isValid = True
            videoIdx = s
            break
        if validStart ^ validStop:
            raise ValueError(f"The reference start ({start}) and reference stop ({stop}) are at different videos!")
    return isValid, videoIdx

def extractVidFramesOfInterest(videoPath, interestMask):
    cap = cv2.VideoCapture(videoPath)
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    maskLen = len(interestMask)
    assert nframes == maskLen, f"The mask of interest (len={maskLen}) does not match the frame number ({nframes}) of the video ({videoPath})."
    frameInds, = np.where(interestMask)
    nInterest = len(frameInds)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameInds[0]-1)   # set to 1st frame of interest
    vidInterest = []
    for i in range(nInterest):
        ret, frame = cap.read()
        vidInterest.append(frame)
        showProgress(i, nInterest, step=500)
    cap.release()
    #cv2.destroyAllWindows()
    return vidInterest

def showProgress(n, nTotal, step):
    progress = (n+1) * 100 / nTotal
    isLast = n + 1 == nTotal
    if n % step == 0 or isLast:
        print("\r", f"{progress:.2f} %", end="")
        
def previewVideoFrame(vidInterests, videoTitles, frameIdx, nRow, nCol, 
                      figWidthInch, figHeightInch):
    nVid = len(vidInterests)
    fig, axes = plt.subplots(nRow, nCol, figsize=(figWidthInch,figHeightInch))
    for i, ax in enumerate(axes.flatten()):
        if i < nVid:
            ax.imshow(vidInterests[i][frameIdx])
            ax.set_title(videoTitles[i])
        ax.axis("off")


#%% Parameters

ttlDir = "D:/MUA_analysis/data/2022-12-19_18-53-46/nidaq/experiment1/recording1/events/NI-DAQmx-100.0/TTL_1_from_analog"
videoDir = "F:/NovDec2022ChR2batch_TL/mouse_13_brown_all/2022-12-19_18-53-46"
saveFilename = "merged_video.avi"
ttlKeys = ["Camera", "platformFB", "platformA", "OptoRight","OptoLeft"]
cameraCh = 1
refCh = 6
refStartTtlIdx = 10#233   # either the idx for refTtl or idx for video frame
nTtl = 1000 #max15000
ttlShifts = -500
videoBreakThreshSec = 5   # threshold duration for breaking the videos in second
vidExtension = ".avi"
nRow = 2
subplotSizeInch = 2
videoTitles = ["Whisker","Right eye", "Left eye", "Top", "Front"]

#%% Get data and preview TTLs

(videoIdx, videoCropMask, videoFrameRate, ttls, uniquePveChs, camTtls, videoStarts, 
 videoEnds, refStart, refStop) = getDataAndPreviewTtls(ttlKeys, cameraCh, refCh, 
            refStartTtlIdx, nTtl, videoBreakThreshSec, ttlShifts, useSec=True)
files = os.listdir(videoDir)
vidFiles = [file for file in files if file.endswith(vidExtension)]
dateTimeStartIdx = vidFiles[0].find("_") + 1
vidDateTime = np.unique([file[dateTimeStartIdx:-len(vidExtension)] for file in vidFiles])
dateTimeInterest = vidDateTime[videoIdx]
vidPrefix = "cam"
vidPostfix = dateTimeInterest + vidExtension
vidFilesMatches = [re.fullmatch(f"{vidPrefix}.*{vidPostfix}", file) for file in vidFiles]
vidFilesInterest = [match[0] for match in vidFilesMatches if match is not None]
vidFilesInterest.sort()
videoSavePath = os.path.join(videoDir, saveFilename)

#%% Extract and preview multi-view video

frameIdx = 0   # for previewing the multi-view video frame
vidInterests = []
for v, vidFile in enumerate(vidFilesInterest):
    print(f"\nExtracting video {v+1}/{len(vidFilesInterest)}")
    videoPath = os.path.join(videoDir, vidFile)
    vidInterest = extractVidFramesOfInterest(videoPath, videoCropMask)
    vidInterests.append(vidInterest)

nFrame = videoCropMask.sum()
frameIntervMsec = 1000 / videoFrameRate
nVid = len(vidInterests)
nCol = int(np.ceil(nVid / nRow))
figWidthInch = nCol * subplotSizeInch
figHeightInch = nRow * subplotSizeInch
previewVideoFrame(vidInterests, videoTitles, frameIdx, nRow, nCol, figWidthInch, figHeightInch)
# tmpPath = os.path.join(videoDir, 'tmp.npy')
# np.save(tmpPath, vidInterests)

#%% Merge and save multi-view video

#tmpPath = os.path.join(videoDir, 'tmp.npy')
#vidInterests = np.load(tmpPath)
writer = FFMpegWriter(fps=videoFrameRate, bitrate=1800)   # bitrate: video quality, the size will be larger for higher quality
fig, axes = plt.subplots(nRow, nCol, figsize=(figWidthInch,figHeightInch))
plt.close(fig)

ims = []
print("Getting multi-view frames...")
for n in range(nFrame):
    showProgress(n, nFrame, step=100)
    multiVidFrame = []
    for i, ax in enumerate(axes.flatten()):
        if i < nVid:
            im = ax.imshow(vidInterests[i][n])
            ax.set_title(videoTitles[i])
            multiVidFrame.append(im)
        ax.axis("off")
    ims.append(multiVidFrame)
    
print("\nSaving video...")
ani = animation.ArtistAnimation(fig, ims, interval=frameIntervMsec, blit=True, repeat_delay=1000)
ani.save(videoSavePath, writer=writer)
print(f"Done! The video is saved at {videoSavePath}.")
# os.remove(tmpPath)

