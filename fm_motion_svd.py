"""
This script uses facemap to compute the motion SVD of all video files found in
the "dir" folder.
"""
import os
from facemap import process
import numpy as np

dir = r"E:\GitHub\Kremkow-Rotation-BinKhalid\data\videos\test"

# Initialise lists of video file paths and names
vid_list = []
vid_names = []

# get list of videos to process
for file in os.scandir(dir):
    vid_list.append([file.path])
    vid_names.append(file.name)

# get SVD per video
for i, vid in enumerate(vid_list):
    # run facemap SVD processing
    savename = process.run([vid], motSVD=True)

    # only save SVD data
    arr = np.load(savename, allow_pickle=True)
    SVDs = arr.flatten()[0]["motSVD"][0]
    fname = os.path.join(
        dir, '.'.join(vid_names[i].split(".")[:-1]) + "_svd.npy"
    )
    np.save(fname, SVDs)
    print("SVDs saved in", fname)

    # remove facemap "_proc.npy" file
    os.remove(savename)
