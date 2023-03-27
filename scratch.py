import numpy as np
import h5py
from settings import proc_dir
import os
import matplotlib.pyplot as plt
import matplotlib as mpl


# colorlist = [
#     'blue',
#     'blue',
#     'blue',
#     'blue',
#     'teal',
#     'teal',
#     'green',
#     'green',
#     'green',
#     'green',
#     'green',
#     'orange',
#     'red',
#     'red',
#     'red'
# ]

colorlist = ["#00A8CC", "#2F80ED", "#4A90E2", "#4FC3F7", "#008080", "#00CED1", '#00FF00', '#80FF00', '#00FF80', '#008040', '#339933', '#FFA500', '#F50C2A', '#FF4D4D', '#D60A0A']

# plt.style.use('seaborn-v0_8-dark-palette')
plt.rcParams.update({'font.size': 16})
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colorlist)


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


#open a h5 file
for file in os.scandir(proc_dir):
    if file.name.endswith(".h5"):
        f = h5py.File(file.path, 'r')
        plt.figure(figsize=(15,8))
        for coord in coord_keys:
            like = np.array(f['Facemap'][coord]['likelihood'])

            plt.plot(like, label=coord)
            xticks = np.arange(0, len(like), 500)
            plt.xticks(np.arange(0, len(like), 200))
            fname = file.name.split(".")[0]

        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # plt.xlim(0, 2700)
        plt.ylabel("Likelihood")
        plt.xlabel("Frame")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig(f"{fname}_L.png")
        plt.close()

        f.close()