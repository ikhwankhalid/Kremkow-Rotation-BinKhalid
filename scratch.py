import numpy as np
import h5py
from settings import rot_dir
import os
import matplotlib.pyplot as plt

#open a h5 file
f = h5py.File(os.path.join(rot_dir, 'cam0_2023-03-21-16-23-18_R_FacemapPose.h5'), 'r')

coord_keys = [
    'eye(back)',
    'eye(bottom)',
    'eye(front)',
    'eye(top)',
    'lowerlip',
    'mouth',
    'nose(bottom)',
    'nose(r)',
    'nose(tip)',
    'nose(top)',
    'nosebridge',
    'paw',
    'whisker(I)',
    'whisker(II)',
    'whisker(III)'
]
data_keys = ['likelihood', 'x', 'y']

like_mouth = np.array(f['Facemap']['eye(top)']['likelihood'])

plt.plot(like_mouth)
xticks = np.arange(0, len(like_mouth), 500)
plt.xticks(np.arange(0, len(like_mouth), 200))
plt.savefig("Haha.png")

f.close()