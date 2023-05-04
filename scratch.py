import os
import numpy as np
import subprocess
from settings import vids_dir, raw_dir, proc_dir
from datetime import datetime
import cv2
import h5py

# dir = r'E:\GitHub\Kremkow-Rotation-BinKhalid\data\videos\Tatiana_behaviorVids_loaderrors'
dir = r"E:\GitHub\Kremkow-Rotation-BinKhalid\data\videos\raw"

###############################################################################
# Script                                                                      #
###############################################################################
for file in os.scandir(dir):
    if file.name.endswith(".h5"):
        f = h5py.File(file.path, 'r')
        whisk1_x = np.array(f['Facemap']['whisker(I)']['x'])
        print(whisk1_x.shape)
