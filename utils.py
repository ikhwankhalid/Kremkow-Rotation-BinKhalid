import numpy as np

def min2s(min, s):
    return min * 60 + s


def min_time(time, min, s):
    return time[np.argmin(np.abs(time - min2s(min, s)))]


def getVideoInfo(
    ttls,
    videoBreakThresh,
    samplingRate        # neuropixel
):
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

    return (
        nVideos,
        videoBreaks,
        videoStarts,
        videoEnds,
        videoDurationsSec,
        videoFrameRate          # video
    )
