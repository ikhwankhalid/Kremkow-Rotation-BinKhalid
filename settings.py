import os

proj_dir = os.path.dirname(os.path.realpath(__file__))
vids_dir = os.path.join(proj_dir, "data", "videos")
raw_dir = os.path.join(vids_dir, "raw")
rot_dir = os.path.join(vids_dir, "rotated")
model_dir = os.path.join(proj_dir, "data", "models")