from settings import vids_dir, raw_dir, rot_dir, model_dir
import os
import facemap
from facemap import process
from facemap.pose.pose import Pose
from glob import glob
import torch
import numpy as np
import imageio

vid_list = []
vid_names = []

model = torch.load(os.path.join(model_dir, "refined_model006.pt"))


for file in os.scandir(rot_dir):
    vid_list.append([file.path])
    vid_names.append(file.name)


os.makedirs(os.path.join(vids_dir, "processed"), exist_ok=True)
for i in range(len(vid_names)):
    poser = Pose(
        filenames=[vid_list[i]],
        net = model,
        model_name = "refined_model006"
    )
    file=vid_list[i][0] #the path of the video
    vid=imageio.get_reader(file, 'ffmpeg')
    totalframes = vid.count_frames()

    print(totalframes)

    poser.load_model()
    poser.pose_prediction_setup()
    # poser.run_all()
    pred, meta = poser.predict_landmarks(0, np.arange(0, totalframes, 1))
    poser.cumframes = [totalframes]
    poser.save_data_to_hdf5(pred.cpu().numpy(), 0, selected_frame_ind=None)





