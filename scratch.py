import os
import numpy as np

dir = r'D:\2022-04-06_16-04-19_Record Node 105\experiment1\recording1\events\Message_Center-904.0\TEXT_group_1'

for file in os.scandir(dir):
    data = np.load(file.path, allow_pickle=True)
    print(file.name)
    print(data.shape)
    print(data)