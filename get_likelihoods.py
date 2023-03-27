import numpy as np
import h5py
from settings import proc_dir, colorlist, coord_keys, data_keys
import os
import matplotlib.pyplot as plt
import matplotlib as mpl


# plt.style.use('seaborn-v0_8-dark-palette')
plt.rcParams.update({'font.size': 18})
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colorlist)


# open a h5 file
for file in os.scandir(proc_dir):
    if file.name.endswith(".h5"):
        f = h5py.File(file.path, 'r')
        plt.figure(figsize=(20, 8))
        for coord in coord_keys:
            like = np.array(f['Facemap'][coord]['likelihood'])

            plt.plot(like, label=coord)
            fname = file.name.split(".")[0]

        xticks = np.arange(0, len(like), int(len(like) / 4))
        plt.xticks(np.round(xticks, -2))
        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # plt.xlim(0, 2700)
        plt.ylabel("Likelihood")
        plt.xlabel("Frame")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig(os.path.join(proc_dir, f"{fname}_L.png"))
        plt.close()

        f.close()
