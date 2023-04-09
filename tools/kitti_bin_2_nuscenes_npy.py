# -*- coding:utf-8 -*-
# author: Awet H. Gebrehiwot
# at 10/13/22
# --------------------------|
import glob
import os
from os.path import join

import numpy as np

from points2spherical2d_corrected import Spherical3DProjection


def parse_calibration(filename):
    """ read calibration file with given filename

        Returns
        -------
        dict
            Calibration matrices as 4x4 numpy arrays.
    """
    calib = {}

    calib_file = open(filename)
    # print(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib


def parse_poses(filename, calibration):
    """ read poses file with per-scan poses from given filename

        Returns
        -------
        list
            list of poses as 4x4 numpy arrays.
    """
    file = open(filename)
    # print(filename)

    poses = []

    Tr = calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)

    for line in file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

    return poses


def convert_bin2npy(data_path, dest_path, lidar_transform):
    sequeses = sorted(os.listdir(data_path))
    # self.times = []
    if lidar_transform:
        # initialize the Class with the source lidar config and target number of laser beams
        spherical_project = Spherical3DProjection(project=True, H=64, W=2048, fov_up=3.0, fov_down=-25.0,
                                                  target_beams=32)

    for seq in sequeses:
        print(seq)
        frames = sorted(glob.glob(os.path.join(data_path, seq, 'velodyne', '*.bin')))
        seq_folder = join(data_path, seq)

        # Read Calib
        calibrations = parse_calibration(join(seq_folder, "calib.txt"))

        # Read times
        # self.times.append(np.loadtxt(join(seq_folder, 'times.txt'), dtype=np.float32))

        # Read poses
        poses_f64 = parse_poses(join(seq_folder, 'poses.txt'), calibrations)
        # self.poses.append([pose.astype(np.float32) for pose in poses_f64])
        poses = [pose.astype(np.float32) for pose in poses_f64]

        lidar_path = os.path.join(dest_path, seq, 'lidar')
        label_path = os.path.join(dest_path, seq, 'labels')
        pose_path = os.path.join(dest_path, seq, 'poses')
        if not os.path.exists(lidar_path):
            os.makedirs(lidar_path)
            os.makedirs(label_path)
            os.makedirs(pose_path)

        for frame, pose in zip(frames, poses):
            if lidar_transform:
                pcl, ss = spherical_project.lidar_transform(frame)
            else:
                pcl = np.fromfile(frame, dtype=np.float32).reshape(-1, 4)
                ss = np.fromfile(frame.replace('velodyne', 'labels').replace('.bin', '.label'), dtype=np.int32)
            new_frame = frame.split('/')[-1].split('.')[0]
            np.save(os.path.join(lidar_path, new_frame), pcl)
            np.save(os.path.join(label_path, new_frame), ss)
            np.save(os.path.join(pose_path, new_frame), pose)


if __name__ == '__main__':
    data_path = '/mnt/personal/gebreawe/Datasets/RealWorld/semantic-kitti/dataset/sequences'
    dest_path = '/mnt/personal/gebreawe/Datasets/RealWorld/semantic-kitti/all_npy_32beam/sequences'
    lidar_transform = True
    convert_bin2npy(data_path, dest_path, lidar_transform)
