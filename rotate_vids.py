import subprocess
import os
from settings import vids_dir, raw_dir, proc_dir

# Set angle to rotate video by
angle = 90

vid_list = []
vid_names = []
os.makedirs(raw_dir, exist_ok=True)


for file in os.scandir(raw_dir):
    vid_list.append(file.path)
    vid_names.append(file.name)


os.makedirs(proc_dir, exist_ok=True)
for vid_name, vid_in in zip(vid_names, vid_list):
    # Name of output file
    video_out = os.path.join(
        vids_dir, "processed",f'{vid_name.split(".")[0]}_R.mp4'
    )

    # Rotate video using ffmpeg
    filter = (
        f"transpose={angle}*PI/180, scale=800:-1, eq=contrast=1.15,"
        + " eq=brightness=0.01"
    )
    subprocess.call(
        [
            'ffmpeg',
            '-i',
            vid_in,
            '-t',
            '00:00:10',
            '-c:v',
            'h264_nvenc',
            '-cq:v',
            '29',
            '-maxrate',
            '100M',
            '-vf',
            filter,
            video_out
        ]
    ) 