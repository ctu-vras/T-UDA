# -*- coding:utf-8 -*-
# author: Awet H. Gebrehiwot
# at 7/8/22
# --------------------------|
import glob
import os

import numpy as np

data_path = 'dataset/semantic-kitti'
list_dir = sorted(os.listdir(data_path))
l = os.listdir(os.path.join(data_path, list_dir[0]))
for c, seq in enumerate(list_dir[11:]):
    frames = sorted(glob.glob(os.path.join(data_path, seq, 'velodyne/*.bin')))
    for id, frame in enumerate(frames):
        with open(frame, 'rb') as b:
            pcl = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        with open(frame.replace('velodyne', 'labels').replace('.bin', '.label'), 'rb') as s:
            ss = np.fromfile(s, dtype=np.int32).reshape(-1)
        if id < 16:
            if not os.path.exists(f"{seq}/lidar"):
                os.makedirs(f"{seq}/lidar")
                os.mkdir(f"{seq}/labels")
            np.save(f"{seq}/lidar/{str(id).zfill(6)}", pcl)
            np.save(f"{seq}/labels/{str(id).zfill(6)}", ss)

    if c > 12:
        break
