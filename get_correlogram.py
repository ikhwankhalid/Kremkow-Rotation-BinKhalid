import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from settings import proc_dir, coord_keys, data_keys
from scipy.signal import butter, lfilter, correlate


def plot_correlogram(signal1, signal2, fps, savename):
    # Calculate the correlogram
    correlogram = correlate(signal1, signal2, mode='full', method='auto')

    # Normalize the correlogram
    correlogram /= np.max(correlogram)

    # Find the maximum value and its corresponding index
    max_value = np.max(correlogram)
    max_index = np.argmax(correlogram)

    # Compute the time lags
    time_lags = np.arange(-len(signal1) + 1, len(signal2))
    time_lags_s = time_lags / 222

    # Plot the correlogram
    fig, ax = plt.subplots()
    ax.plot(time_lags_s, correlogram)
    ax.set_xlabel('Time lag (seconds)')
    ax.set_ylabel('Normalized correlation')
    ax.set_title('Correlogram')

    # Mark the maximum value
    ax.plot(time_lags_s[max_index], max_value, 'ro')

    # # Create an inset with a zoomed view of the maximum value
    # ax_inset = plt.axes([0.55, 0.55, 0.3, 0.3])
    # ax_inset.plot(time_lags, correlogram)
    # ax_inset.set_xlim(time_lags[max_index] - 10, time_lags[max_index] + 10)
    # ax_inset.set_ylim(0.9, 1.1)
    # ax_inset.set_title('Zoomed Inset')

    print(f"Max at {time_lags_s[max_index]} seconds.")

    plt.savefig(f"{savename}.png")


def low_pass_filter(data, cutoff_frequency, sample_frequency, order=1):
    """
    Apply a low-pass filter to time series data to remove high-frequency noise.

    Parameters:
    data (list or np.array): The time series data to be filtered.
    cutoff_frequency (float): The cutoff frequency of the low-pass filter. All 
                                frequencies higher than this will be filtered 
                                out.
    sample_frequency (float): The sampling frequency of the time series data.
    order (int, optional): The order of the Butterworth filter. Default is 1.

    Returns:
    filtered_data (np.array): The filtered time series data.
    """

    # Calculate the Nyquist frequency
    nyquist_frequency = 0.5 * sample_frequency

    # Normalize the cutoff frequency with respect to the Nyquist frequency
    normalized_cutoff_frequency = cutoff_frequency / nyquist_frequency

    # Design the low-pass Butterworth filter
    b, a = butter(order, normalized_cutoff_frequency,
                  btype='low', analog=False)

    # Apply the filter to the data
    filtered_data = lfilter(b, a, data)

    return filtered_data


if __name__ == "__main__":
    # signal1 = np.sin(np.linspace(0, 10, 1000))
    # signal2 = np.sin(np.linspace(0, 10, 1000) + np.pi/2)
    # plot_correlogram(signal1, signal2, "correlogram")

    for file in os.scandir(proc_dir):
        if file.name.endswith(".h5"):
            f = h5py.File(file.path, 'r')
            vidname = file.path.split("_")

            trial = "_".join(file.path.split("_")[:2])
            proc_dict = np.load(f"{trial}_proc.npy", allow_pickle=True).item()
            motSVD = np.array(proc_dict["motSVD"])[0]

            print(motSVD.shape)

            plt.figure(figsize=(15, 10))
            plt.plot(motSVD[:3000, :5])
            plt.savefig("motSVD.png")
            plt.close()

            # plt.figure(figsize=(20, 8))
            # for coord in coord_keys:
            #     like = np.array(f['Facemap'][coord]['likelihood'])

            whisk1_x = np.array(f['Facemap']['whisker(I)']['x'])
            whisk1_y = np.array(f['Facemap']['whisker(I)']['y'])

            # numb = 5000
            # numb2 = numb + 800

            whisk1_x = np.array(f['Facemap']['whisker(I)']['x'])
            whisk1_y = np.array(f['Facemap']['whisker(I)']['y'])

            paw_x = np.array(f['Facemap']['paw']['x'])
            paw_y = np.array(f['Facemap']['paw']['y'])

            whisk1_like = np.array(f['Facemap']['whisker(I)']['likelihood'])
            paw_like = np.array(f['Facemap']['paw']['likelihood'])

            # whisk1_x = whisk1_x[paw_like>0.9]
            # paw_x = paw_x[paw_like>0.9]

            # whisk1_x = low_pass_filter(whisk1_x, 1, 200)
            # paw_x = low_pass_filter(paw_x, 1, 200)

            # whisk1_x = np.diff(whisk1_x)
            # paw_x = np.diff(paw_x)

            whisk1_x = whisk1_x - np.mean(whisk1_x)
            paw_x = paw_x - np.mean(paw_x)

            whisk1_y = whisk1_y - np.mean(whisk1_y)
            paw_y = paw_y - np.mean(paw_y)

            plt.figure(figsize=(15, 10))
            plt.plot(whisk1_x)
            plt.plot(paw_x)
            plt.savefig("Trajec.png")
            plt.close()

            plt.figure(figsize=(15, 10))
            plt.plot(whisk1_like)
            plt.savefig("Likelihood")
            plt.close()

            plot_correlogram(whisk1_x, paw_x, 222, "correlogram")

            f.close()
