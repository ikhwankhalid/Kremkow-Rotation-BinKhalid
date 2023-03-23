# from facemap import process
from glob import glob
from moviepy.editor import *
import os
import multiprocessing

num_threads = multiprocessing.cpu_count()
fps = 30

simultaneous_vid_list = []
proj_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(proj_dir, "data")
os.makedirs(os.path.join(data_dir, "raw"), exist_ok=True)

for file in os.scandir(os.path.join(data_dir, "raw")):
    filename = file.path
    simultaneous_vid_list.append(filename)

print(simultaneous_vid_list)

for vidname in simultaneous_vid_list:
    clip = VideoFileClip(vidname)
    clip = clip.subclip(120)
    clip = clip.rotate(105)

    # width_of_clip2 = clip_resized.w
    # height_of_clip2 = clip_resized.h

    # clip.ipython_display(width=480)
    print(vidname)
    clip.write_videofile(
        f"vid-R.mp4", threads=num_threads, fps=fps
    )
