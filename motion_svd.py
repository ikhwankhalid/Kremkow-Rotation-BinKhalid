import os
import numpy as np
import subprocess
from settings import vids_dir, raw_dir, proc_dir
from datetime import datetime
import h5py
import cv2
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from functools import partial
from tqdm import tqdm


# dir = r'E:\GitHub\Kremkow-Rotation-BinKhalid\data\videos\Tatiana_behaviorVids_loaderrors'
# dir = r"E:\GitHub\Kremkow-Rotation-BinKhalid\data\videos\raw"
dir = r"E:\GitHub\Kremkow-Rotation-BinKhalid\data\videos\minicut"

# fname = os.path.join(dir, "cam3_2023-05-16-15-41-57.avi")
fname = os.path.join(dir, "cam4_2022-04-06-16-20-35_786.0.mp4")


def process_chunk(frames, num_svds):
    svd_elements = []
    
    for idx, frame in enumerate(frames[:-1]):
        next_frame = frames[idx + 1]

        # Compute the difference between subsequent frames
        diff = np.abs(next_frame.astype(np.int16) - frame.astype(np.int16))

        try:
            # Perform SVD on the difference image
            U, s, V = np.linalg.svd(diff)
            svd_elements.append(s[:num_svds])

        # If the difference is zero, use the previous SVD value
        except np.linalg.LinAlgError:
            svd_elements.append(svd_elements[-1])  

    return svd_elements


def perform_motion_svds(video_file, num_svds, chunk_size=10000, n_jobs=-1):
    # Load video
    video = cv2.VideoCapture(video_file)

    # Get total number of frames
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize a buffer for the frames
    frame_buffer = []

    # Read the video in chunks
    idx = 0
    while idx < total_frames:
        # Read frames and store them in the buffer
        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_buffer.append(frame)
        idx += 1

        # If the buffer is full, process the chunk
        if len(frame_buffer) == chunk_size:
            # Remove the first frame to overlap the chunks for continuous processing
            frame_buffer.pop(0)

    video.release()

    # Determine the number of chunks for parallelization
    num_chunks = (len(frame_buffer) + chunk_size - 1) // chunk_size

    # Divide the frame_buffer into chunks for parallel processing
    chunks = [frame_buffer[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

    # Process chunks in parallel using Joblib
    results = Parallel(
        n_jobs=n_jobs
    )(delayed(process_chunk)(chunk, num_svds) for chunk in chunks)

    # Combine the results
    svd_elements = []
    for result in results:
        svd_elements.extend(result)

    return svd_elements


num_svds = 5

# Perform motion SVD on the video and store the values in a list
svs = perform_motion_svds(fname, num_svds)


# Plot the first 5 singular values over time
plt.figure()
plt.plot(svs)
plt.ylabel('Singular Values')
plt.xlabel('Frame Number')
plt.legend(['SV1', 'SV2', 'SV3', 'SV4', 'SV5'])
plt.show()
plt.close()

print(svs)
