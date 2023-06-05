"""
This script uses facemap to compute the motion SVD of all video files found in
the "dir" folder.
"""
import os
from facemap import process

dir = r"E:\GitHub\Kremkow-Rotation-BinKhalid\data\videos\test"

# Initialise lists of video file paths and names
vid_list = []
vid_names = []

# get list of videos to process
for file in os.scandir(dir):
    print(file.path)
    vid_list.append([file.path])
    vid_names.append(file.name)

# get SVDs
savename = process.run(vid_list, motSVD=True)
print("Output saved in", savename)
