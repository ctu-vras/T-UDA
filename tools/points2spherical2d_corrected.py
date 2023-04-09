# -*- coding:utf-8 -*-
# author: Awet H. Gebrehiwot
# at 9/2/22
# --------------------------|
# !/usr/bin/env python3
import copy
import math
import os.path

import numpy as np
# import cv2
# import pptk


class Spherical3DProjection:
    """Class that contains LaserScan with x,y,z,r"""
    EXTENSIONS_SCAN = ['.bin', '.npy']

    def __init__(self, project=True, H=64, W=2048, fov_up=3.0, fov_down=-25.0, target_beams=32):
        # def __init__(self, project=True, H=64, W=1024, fov_up=3.0, fov_down=-25.0):
        self.project = project
        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.target_beams = target_beams
        self.reset()

    def reset(self):
        """ Reset scan members. """
        self.points = np.zeros((0, 3), dtype=np.float32)  # [m, 3]: x, y, z

        # projected range image - [H,W] range (-1 is no data)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)

        # projected labels - [H,W] class labels (-1 is no data)
        self.proj_labels = np.full((self.proj_H, self.proj_W), -1,
                                   dtype=np.float32)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                                dtype=np.int32)

        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.float32)  # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.float32)  # [m, 1]: y

        # mask containing for each pixel, if it contains a point or not
        self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                                  dtype=np.int32)  # [H,W] mask

    def size(self):
        """ Return the size of the point cloud. """
        return self.points.shape[0]

    def __len__(self):
        return self.size()

    def lidar_transform(self, filename, dataset='Kitti'):
        """ Open raw scan and fill in attributes
    """
        # reset just in case there was an open structure
        self.reset()

        # check filename is string
        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(filename))))

        # check extension is a laserscan
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
            raise RuntimeError("Filename extension is not valid scan file.")
        labels = None
        # if all goes well, open pointcloud
        if dataset == "Kitti" or dataset == "kitti" or dataset == "SemanticKitti":
            scan = np.fromfile(filename, dtype=np.float32)
            scan = scan.reshape((-1, 4))
            # put in attribute
            points = scan #[:, 0:3]  # get xyz
            if not os.path.exists(filename):
                raise RuntimeError("Labels path does not exist")
            labels = np.fromfile(filename.replace("velodyne", "labels").replace(".bin", ".label"), dtype=np.int32)
        elif dataset == "WOD" or dataset == "wod" or dataset == "Wod":
            scan = np.load(filename)
            # put in attribute
            points = scan  # scan[:, 0:3]  # get xyz

            if not os.path.exists(filename):
                raise RuntimeError("Labels path does not exist")
            labels = np.load(filename.replace("lidar", "labels"))
            if len(labels.shape) == 2:
                if labels.shape[1] == 2:
                    labels = labels[:, 1]
        labels = labels & 0xFFFF

        self.set_points(points, labels)

        transformed_points, transformed_labels = None, None

        # if projection is wanted, then do it and fill in the structure
        if self.project:
            transformed_points, transformed_labels = self.do_range_projection()
            # if self.save:
            #     frame = filename.split("/")[-1].split(".")[0]
            #     frame = str(frame).zfill(6)

        return transformed_points, transformed_labels

    def set_points(self, points, labels=None):
        """ Set scan attributes (instead of opening from file)
    """
        # reset just in case there was an open structure
        self.reset()

        # check scan makes sense
        if not isinstance(points, np.ndarray):
            raise TypeError("Scan should be numpy array")

        # put in attribute
        self.points = points  # get xyz

        if labels is not None:
            self.labels = labels  # get remission
        else:
            self.labels = np.zeros((points.shape[0]), dtype=np.float32)

    # get reduced lidar beam e.g. return 32 beam index lidar
    def get_reduced_lidar_beam(self, points_indexing_top, lidar_beam=32):
        lidar_index_fs = int(self.proj_H / lidar_beam)  # 64 /32 --> 2
        lidar_beam_index_mask = np.zeros(points_indexing_top.shape[0]).astype(bool)
        start_ind = 0
        for i in range(self.proj_H):

            if i % lidar_index_fs == 0:
                lidar_beam_index_mask[points_indexing_top == i] = True
        return lidar_beam_index_mask

    def do_range_projection(self):
        """ Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
    """
        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi  # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # pptk.viewer(self.points[:, :3], self.labels)

        # TODO: check this rotation if it is required
        # rotate 180 degree to correct the start of the scanning
        rotate_rad = np.deg2rad(180)
        c, s = np.cos(rotate_rad), np.sin(rotate_rad)
        j = np.matrix([[c, s], [-s, c]])
        self.points[:, :2] = np.dot(self.points[:, :2], j)

        # get depth of all points
        depth = np.linalg.norm(self.points[:, :3], 2, axis=1)

        # get scan components
        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]

        # get angles of all points
        # yaw = []
        # for y, x in zip(scan_y, scan_x):
        #     if y > 0:
        #         yaw.append(-np.arctan2(y, x))
        #     else:
        #         yaw.append(2 * np.pi - np.arctan2(y, x))

        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        # for i in range(self.points.shape[0] // 2048):
        #     print("", np.mean(proj_y[i * 2048:(i + 1) * 2048]), np.std(proj_y[i * 2048:(i + 1) * 2048]),
        #           proj_y[i * 2048:(i + 1) * 2048])


        # scale to image size using angular resolution
        proj_x *= self.proj_W  # in [0.0, W]
        proj_y *= self.proj_H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)  # np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

        proj_y = np.floor(proj_y)  # np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        points = self.points  # [order]
        labels = self.labels  # [order]
        proj_y = proj_y  # [order]
        proj_x = proj_x  # [order]

        # TODO: Check if this works in all lidar data
        """
        This block of code is used to refine the incorrectly projected point clouds into the 2d spherical view.
        It fix the pitch angel of each point based on their scanning order 
        and also change in the yaw angle between consecutive points (order of storing).
        """
        bines = np.zeros_like(self.points[:, 0])
        # print(proj_x, proj_y)
        count = 0
        j = 0
        row = 0
        deltas = []
        for i in range(self.points.shape[0]):
            count += 1
            x_delta = proj_x[i - j] - proj_x[i]
            deltas.append(x_delta)
            if x_delta >= - self.proj_H:
                bines[i] = row
            # elif count >= self.proj_H:
            #     count = 0
            #     row += 1
            #     bines[i] = row
            else:
                count = 0
                row += 1
                bines[i] = row
            j = 1
        ###############################

        # num_points = points.shape[0]
        # colors = np.array([[1, 0, 0]] * num_points)
        # colors[0:2048, :] = (0, 1, 0)

        # for i in range(self.proj_H):
        #     colors[proj_y == i] = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 1)][i % 6]
        #     # print("row ", np.mean(proj_y[proj_y == i]), np.std(proj_y[proj_y == i]), proj_y[proj_y == i])
        # 
        # colors2 = np.array([[1, 0, 0]] * num_points)
        # colors2[0:2048, :] = (0, 1, 0)
        # 
        # for i in range(self.proj_H):
        #     colors2[bines == i] = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 1)][i % 6]
        # pptk.viewer(points, colors)
        # pptk.viewer(points, colors2)
        #
        # compressed_proj_ = self.get_reduced_lidar_beam(proj_y)
        # comp_points = points[compressed_proj_]
        # comp_labels = labels[compressed_proj_]
        # pptk.viewer(comp_points, comp_labels)

        compressed_proj_ = self.get_reduced_lidar_beam(bines, self.target_beams)
        comp_points = points[compressed_proj_]
        comp_labels = labels[compressed_proj_]
        # pptk.viewer(comp_points, comp_labels)

        # rotate back by 180 degree to transform it into the original orinetation
        rotate_rad = np.deg2rad(180)
        c, s = np.cos(rotate_rad), np.sin(rotate_rad)
        j = np.matrix([[c, s], [-s, c]])
        comp_points[:, :2] = np.dot(comp_points[:, :2], j)
        # pptk.viewer(comp_points, comp_labels)

        return comp_points, comp_labels

# # filename = "/home/success/Documents/rciServer/mnt/personal/gebreawe/Datasets/RealWorld/semantic-kitti/dataset/sequences/08/velodyne/000300.bin"
# filename = "/home/success/Documents/PhD/code/lidar_transfer/minimal/sequences/00/velodyne/000000.bin"
# spherical_project = Spherical3DProjection()
# spherical_project.lidar_transform(filename)

