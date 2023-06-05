"""
This script computes the movie SVD of the video file specified by "fname"
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

dir = r"E:\GitHub\Kremkow-Rotation-BinKhalid\data\videos\minicut"
fname = os.path.join(dir, "cam4_2022-04-06-16-20-35_786.0.mp4")


def movie_svd_per_frame(frame, n_singular_values):
    # Perform SVD
    try:
        U, singular_values, Vt = np.linalg.svd(frame, full_matrices=False)

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


def movie_svd(video_path, n_singular_values=10, n_jobs=-1, chunk_size=20):
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

        # Calculate movie SVD in parallel
        trunc_singular_values_list = Parallel(
            n_jobs=n_jobs
        )(delayed(
            movie_svd_per_frame
        )(frames[i], n_singular_values) for i in range(len(frames)))

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

    singular_values = movie_svd(
        fname, n_singular_values, n_jobs, chunk_size
    )

    plt.figure()
    plt.plot(singular_values)
    plt.show()
