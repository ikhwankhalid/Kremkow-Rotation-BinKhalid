import os
import numpy as np
import subprocess
from settings import vids_dir, raw_dir, proc_dir
from datetime import datetime
import h5py
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
plt.rcParams.update({'font.size': 20})


def rolling_variance(arr, window_width):
    if window_width < 1:
        raise ValueError("Window width must be a positive integer.")

    if window_width > len(arr):
        raise ValueError(
            "Window width must be smaller or equal to the array length."
        )

    variances = []
    for i in range(len(arr) - window_width + 1):
        window = arr[i: i + window_width]
        variances.append(np.var(window))

    out = np.array(variances)
    out = np.pad(out, (window_width // 2, 0), 'edge')

    return out


def filter_frequency(array, low_freq, high_freq, sampling_rate):
    # Calculate the Fourier Transform
    fourier_transform = fft(array)

    # Calculate the frequencies
    frequencies = np.fft.fftfreq(len(array), 1 / sampling_rate)

    # Filter out components outside the frequency range
    filtered_transform = fourier_transform.copy()
    filtered_transform[(frequencies < low_freq) | (frequencies > high_freq)] = 0

    # Calculate the inverse Fourier Transform
    filtered_array = ifft(filtered_transform)

    return filtered_array


sample_rate = 222
lo_freq = 9
hi_freq = 25
window_width = 300


###############################################################################
# Script                                                                      #
###############################################################################
if __name__ == '__main__':
    if not os.path.exists(proc_dir):
        raise Exception("'processed' directory does not exist!")
    # Initialise lists of video file paths and names
    file_paths = []
    file_names = []
    # os.makedirs(dir, exist_ok=True)

    # Get list of all videos in the "raw" folder
    for file in os.scandir(proc_dir):
        if file.name.endswith(".h5"):
            file_paths.append(file.path)
            file_names.append(file.name)

    if not file_paths:
        raise Exception("No .h5 files found!")

    for fname, fpath in zip(file_names, file_paths):
        f = h5py.File(fpath, 'r')

        whiskx = np.array([
                f['Facemap']['whisker(I)']['x'],
                f['Facemap']['whisker(II)']['x'],
                f['Facemap']['whisker(III)']['x']
            ]
        )

        whisky = np.array([
                f['Facemap']['whisker(I)']['y'],
                f['Facemap']['whisker(II)']['y'],
                f['Facemap']['whisker(III)']['y']
            ]
        )

        whiskx_mean = np.mean(whiskx, axis=0)
        whiskx_mean -= np.mean(whiskx_mean)

        whiskx_hifreq = filter_frequency(
            whiskx_mean, lo_freq, hi_freq, sample_rate
        )
        whiskx_var = rolling_variance(whiskx_hifreq, window_width=60)

        idxs = [67000, 72000]
        tdiff = idxs[1] - idxs[0]
        tplot = np.linspace(0, tdiff, 9)

        # whiskx_var /= np.amax(whiskx_var[idxs[0]:idxs[1]])
        # whiskx_hifreq /= np.amax(whiskx_hifreq[idxs[0]:idxs[1]])
        # whiskx_mean /= np.amax(whiskx_mean[idxs[0]:idxs[1]])

        # [idxs[0]:idxs[1]]

        fig, ax = plt.subplots(3, 1, figsize=(22, 12))
        ax[0].plot(whiskx_mean[idxs[0]:idxs[1]], label="whisker x")
        # ax[0].xticks(tplot, np.round(tplot / sample_rate, 0))
        ax[0].set_xlabel("Frame")
        ax[0].set_title("Mean whisker x (across 3 whisker keypoints)")
        ax[0].set_ylabel("x")

        ax[1].plot(whiskx_hifreq[idxs[0]:idxs[1]], label=r"whisker x (freq $\in$"+f" [{lo_freq}, {hi_freq}] Hz)")
        # ax[0].xticks(tplot, np.round(tplot / sample_rate, 0))
        ax[1].set_xlabel("Frame")
        ax[1].set_title(r"Whisking(?) frequency component (f $\in$"+f" [{lo_freq}, {hi_freq}] Hz)")
        ax[1].set_ylabel("x")

        ax[2].plot(whiskx_var[idxs[0]:idxs[1]], label="variance (sliding window)")
        # ax[0].xticks(tplot, np.round(tplot / sample_rate, 0))
        ax[2].set_xlabel("Frame")
        ax[2].set_title("Variance (sliding window)")
        ax[2].set_ylabel("Variance")

        for ax in fig.get_axes():
            ax.label_outer()

        # plt.legend()
        # plt.savefig(os.path.join(proc_dir, "whiskx_var.png"))
        plt.show()
        plt.close()

        print(whiskx_var)
