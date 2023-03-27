from settings import vids_dir, raw_dir, proc_dir, model_dir
import os
import facemap
from facemap import process
from facemap.pose.pose import Pose
from glob import glob
import torch
import numpy as np
import imageio


###############################################################################
# Parameters                                                                  #
###############################################################################
modelname = "refined_model_EYEVAL"
batch_size = 3


###############################################################################
# Script                                                                      #
###############################################################################
# Initialise lists of video file paths and names
vid_list = []
vid_names = []


# load model fine tuned to lab data
model = torch.load(os.path.join(model_dir, f"{modelname}.pt"))


# get list of videos to process
for file in os.scandir(proc_dir):
    vid_list.append([file.path])
    vid_names.append(file.name)


os.makedirs(os.path.join(vids_dir, "processed"), exist_ok=True)


# run pose prediction on each video in folder
for i in range(len(vid_names)):
    poser = Pose(
        filenames=[vid_list[i]],
        net=model,
        model_name=f"{modelname}"
    )

    # get number of frames in video
    file = vid_list[i][0]  # the path of the video
    vid = imageio.get_reader(file, 'ffmpeg')
    totalframes = vid.count_frames()

    poser.load_model()
    poser.pose_prediction_setup()
    # poser.run_all()
    poser.batch_size = batch_size
    pred, meta = poser.predict_landmarks(0, np.arange(0, totalframes, 1))
    poser.cumframes = [totalframes]
    poser.save_data_to_hdf5(pred.cpu().numpy(), 0, selected_frame_ind=None)
