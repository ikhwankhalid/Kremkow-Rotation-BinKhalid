"""
This script computes the motion SVD of the video file specified by "fname"
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# dir = r"E:\GitHub\Kremkow-Rotation-BinKhalid\data\videos\raw"
# fname = os.path.join(dir, "mouse_face.mp4")

dir = r"E:\GitHub\Kremkow-Rotation-BinKhalid\data\videos\archive"
fname = os.path.join(dir, "cam0_2023-03-22-13-33-33_R.mp4")


def motion_svd_between_frames(frame1, frame2, n_singular_values):
    # Calculate the absolute difference between the two frames
    diff_frame = np.abs(frame1.astype(np.float32) - frame2.astype(np.float32))

    # Perform SVD
    try:
        U, singular_values, Vt = np.linalg.svd(diff_frame, full_matrices=False)

        # Truncate singular values
        trunc_singular_values = np.zeros(n_singular_values)
        n_found_singular_values = min(n_singular_values, len(singular_values))
        trunc_singular_values[
            :n_found_singular_values
        ] = singular_values[
            :n_found_singular_values
        ]

        return trunc_singular_values

    except np.linalg.LinAlgError:
        # If the SVD does not converge, return zero singular values
        return np.zeros(n_singular_values)


def motion_svd(video_path, n_singular_values=10, n_jobs=-1, chunk_size=1000):
    # Load video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_truncated_singular_values = []

    for start_frame in range(0, total_frames, chunk_size):
        # Load frames
        frames = []
        for i in range(chunk_size):
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            else:
                break

        # Calculate motion SVD between frames in parallel
        trunc_singular_values_list = Parallel(
            n_jobs=n_jobs
        )(
            delayed(
                motion_svd_between_frames
            )(
                frames[i], frames[i+1], n_singular_values
            ) for i in range(len(frames) - 1))
        all_truncated_singular_values.extend(trunc_singular_values_list)

        # Check if reached the end of the video
        if len(frames) < chunk_size:
            break

    # Release VideoCapture object
    cap.release()

    # Return singular values as numpy array
    return np.array(all_truncated_singular_values)


if __name__ == "__main__":
    n_singular_values = 10
    n_jobs = -1
    chunk_size = 10000

    singular_values = motion_svd(fname, n_singular_values, n_jobs, chunk_size)

    plt.figure()
    plt.plot(singular_values)
    plt.xlabel("Frame")
    plt.ylabel("Singular value")
    plt.show()
