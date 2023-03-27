import subprocess
import os
from settings import vids_dir, raw_dir, proc_dir

###############################################################################
# Parameters                                                                  #
###############################################################################
# Set angle to rotate video by
angle = 90


###############################################################################
# Script                                                                      #
###############################################################################
# Initialise lists of video file paths and names
vid_list = []
vid_names = []
os.makedirs(raw_dir, exist_ok=True)


# Get list of all videos in the "raw" folder
for file in os.scandir(raw_dir):
    vid_list.append(file.path)
    vid_names.append(file.name)


# Create "processed" folder
os.makedirs(proc_dir, exist_ok=True)


# Rotate each video and re-encode to mp4 format for compression
for vid_name, vid_in in zip(vid_names, vid_list):
    # Name of output file
    video_out = os.path.join(
        vids_dir, "processed", f'{vid_name.split(".")[0]}_R.mp4'
    )

    # Rotate and re-encode video using ffmpeg
    filter = (
        f"transpose={angle}*PI/180, scale=800:-1, eq=contrast=1.15,"
        + " eq=brightness=0.01"
    )
    subprocess.call(
        [
            'ffmpeg',
            # '-accurate_seek',
            # '-ss',
            # '00:35:00',
            '-t',
            '00:00:10',
            '-i',
            vid_in,
            '-c:v',
            'h264_nvenc',           # GPU encoder
            '-cq:v',                # Constant quality
            '29',
            '-maxrate',
            '100M',
            '-vf',
            filter,
            '-avoid_negative_ts',   # Avoid empty frames
            'make_zero',
            video_out
        ]
    )
