from facemap import process
from glob import glob
import os

simultaneous_vid_list = []
proj_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(proj_dir, "data")
os.makedirs(data_dir, exist_ok=True)

for file in os.scandir(data_dir):
    filename = file.path
    simultaneous_vid_list.append(filename)

print(simultaneous_vid_list)
