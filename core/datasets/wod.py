import os
import os.path
from os.path import exists

import numpy as np
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize

from core.datasets.wod_datalist import wod_train, wod_val, wod_test

__all__ = ['WOD']

label_name_mapping = {
    0: "Undefined",
    1: "Car",
    2: "Truck",
    3: "Bus",
    4: "Other Vehicle",
    # Other small vehicles (e.g. pedicab) and large vehicles (e.g. construction vehicles, RV, limo, tram).
    5: "Motorcyclist",
    6: "Bicyclist",
    7: "Pedestrian",
    8: "Sign",
    9: "Traffic Light",
    10: "Pole",  # Lamp post, traffic sign pole etc.
    11: "Construction Cone",  # Construction cone/pole.
    12: "Bicycle",
    13: "Motorcycle",
    14: "Building",
    15: "Vegetation",  # Bushes, tree branches, tall grasses, flowers etc.
    16: "Tree Trunk",
    17: "Curb",  # Curb on the edge of roads. This does not include road boundaries if there’s no curb.
    18: "Road",  # Surface a vehicle could drive on. This include the driveway connecting
    # // parking lot and road over a section of sidewalk.
    19: "Lane Marker",  # Marking on the road that’s specifically for defining lanes such as
    # // single/double white/yellow lines.
    20: "Other Ground",  # Marking on the road other than lane markers, bumps, cateyes, railtracks // etc.
    21: "Walkable",  # Most horizontal surface that’s not drivable, e.g. grassy hill, // pedestrian walkway stairs etc.
    22: "Sidewalk"  # Nicely paved walkable surface when pedestrians most likely to walk on.
}

kept_labels = [
    "Car", "Truck", "Bus", "Other Vehicle", "Motorcyclist", "Bicyclist",
    "Pedestrian", "Sign", "Traffic Light", "Pole", "Construction Cone", "Bicycle",
    "Motorcycle", "Building", "Vegetation", "Tree Trunk", "Curb", "Road",
    "Lane Marker", "Other Ground", "Walkable", "Sidewalk"
]


# label_name_mapping = {
#     0: 'unlabeled',
#     1: 'outlier',
#     10: 'car',
#     11: 'bicycle',
#     13: 'bus',
#     15: 'motorcycle',
#     16: 'on-rails',
#     18: 'truck',
#     20: 'other-vehicle',
#     30: 'person',
#     31: 'bicyclist',
#     32: 'motorcyclist',
#     40: 'road',
#     44: 'parking',
#     48: 'sidewalk',
#     49: 'other-ground',
#     50: 'building',
#     51: 'fence',
#     52: 'other-structure',
#     60: 'lane-marking',
#     70: 'vegetation',
#     71: 'trunk',
#     72: 'terrain',
#     80: 'pole',
#     81: 'traffic-sign',
#     99: 'other-object',
#     252: 'moving-car',
#     253: 'moving-bicyclist',
#     254: 'moving-person',
#     255: 'moving-motorcyclist',
#     256: 'moving-on-rails',
#     257: 'moving-bus',
#     258: 'moving-truck',
#     259: 'moving-other-vehicle'
# }
#
# kept_labels = [
#     'road', 'sidewalk', 'parking', 'other-ground', 'building', 'car', 'truck',
#     'bicycle', 'motorcycle', 'other-vehicle', 'vegetation', 'trunk', 'terrain',
#     'person', 'bicyclist', 'motorcyclist', 'fence', 'pole', 'traffic-sign'
# ]

class WOD(dict):

    def __init__(self, root, voxel_size, num_points, model_configs, **kwargs):
        submit_to_server = kwargs.get('submit', False)
        sample_stride = kwargs.get('sample_stride', 1)
        google_mode = kwargs.get('google_mode', False)
        ssl_mode = kwargs.get('ssl_mode', False)

        if submit_to_server:
            super().__init__({
                'train':
                    WODInternal(root,
                                voxel_size,
                                num_points,
                                sample_stride=1,
                                split='train',
                                submit=True,
                                model_configs=model_configs),
                'test':
                    WODInternal(root,
                                voxel_size,
                                num_points,
                                sample_stride=1,
                                split='test',
                                model_configs=model_configs)
            })
        else:
            super().__init__({
                'train':
                    WODInternal(root,
                                voxel_size,
                                num_points,
                                sample_stride=1,
                                split='train',
                                model_configs=model_configs,
                                google_mode=google_mode,
                                ssl_mode=ssl_mode),
                'test':
                    WODInternal(root,
                                voxel_size,
                                num_points,
                                sample_stride=sample_stride,
                                split='val',
                                model_configs=model_configs)
            })


class WODInternal:

    def __init__(self,
                 root,
                 voxel_size,
                 num_points,
                 split,
                 model_configs,
                 sample_stride=1,
                 submit=False,
                 google_mode=False,
                 ssl_mode=False):
        if submit:
            trainval = True
        else:
            trainval = False
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.sample_stride = sample_stride
        self.google_mode = google_mode
        self.ssl_mode = ssl_mode
        self.source = model_configs.source
        self.target = model_configs.target
        self.seqs = []

        # if split == 'train':
        #     self.seqs = [
        #         '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'
        #     ]
        #
        #     # SSL method added
        #     if self.ssl_mode:
        #         self.seqs += [
        #             '00', '01', '02', '03', '04', '05', '06', '07', '09', '10'
        #         ]
        #         print(f"{split}_loaded datasets {self.seqs} ")
        if split == 'train':
            self.seqs = wod_train
            if self.google_mode or trainval:
                self.seqs.append(wod_val)
        elif self.split == 'val':
            self.seqs = wod_val
        elif self.split == 'test':
            self.seqs = wod_test

        self.files = []
        for seq in self.seqs:
            seq_files = sorted(
                os.listdir(os.path.join(self.root, seq, 'lidar')))
            seq_files = [
                os.path.join(self.root, seq, 'lidar', x) for x in seq_files
            ]
            self.files.extend(seq_files)

        if self.sample_stride > 1:
            self.files = self.files[::self.sample_stride]

        # reverse_label_name_mapping = {}
        # self.npy_map = np.zeros(260)   # (int(model_configs.data.num_classes))
        # cnt = 0
        # for label_id in label_name_mapping:
        #     # if label_id > 250:
        #     #     if label_name_mapping[label_id].replace('moving-',
        #     #                                             '') in kept_labels:
        #     #         self.npy_map[label_id] = reverse_label_name_mapping[
        #     #             label_name_mapping[label_id].replace('moving-', '')]
        #     #     else:
        #     #         self.npy_map[label_id] = 255
        #     # # elif label_id == 0:
        #     # #     self.npy_map[label_id] = 255
        #     # # else:
        #     if label_id == 0:
        #         self.npy_map[label_id] = 255
        #     else:
        #         if label_name_mapping[label_id] in kept_labels:
        #             self.npy_map[label_id] = cnt
        #             reverse_label_name_mapping[
        #                 label_name_mapping[label_id]] = cnt
        #             cnt += 1
        #         else:
        #             self.npy_map[label_id] = 255
        #
        # self.reverse_label_name_mapping = reverse_label_name_mapping
        # self.num_classes = cnt
        # self.angle = 0.0

        reverse_label_name_mapping = {}
        self.label_map = np.zeros(260)
        cnt = 0
        for label_id in label_name_mapping:
            if label_id > 250:
                if label_name_mapping[label_id].replace('moving-',
                                                        '') in kept_labels:
                    self.label_map[label_id] = reverse_label_name_mapping[
                        label_name_mapping[label_id].replace('moving-', '')]
                else:
                    self.label_map[label_id] = 255
            elif label_id == 0:
                self.label_map[label_id] = 255
            else:
                if label_name_mapping[label_id] in kept_labels:
                    self.label_map[label_id] = cnt
                    reverse_label_name_mapping[
                        label_name_mapping[label_id]] = cnt
                    cnt += 1
                else:
                    self.label_map[label_id] = 255

        self.reverse_label_name_mapping = reverse_label_name_mapping
        self.num_classes = cnt
        self.angle = 0.0

    def set_angle(self, angle):
        self.angle = angle

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # with open(self.files[index], 'rb') as b:
        #     block_ = np.load(b).reshape(-1, 4)
        block_ = np.load(self.files[index])[:, :4].astype(np.float32)  # .reshape(-1, 4)
        block_[:, 2] = block_[:, 2] - 2  # translate the z coordinate by -2 meters to make wod similar with the
        # semantic-kitti data
        block = np.zeros_like(block_)

        if 'train' in self.split:
            theta = np.random.uniform(0, 2 * np.pi)
            scale_factor = np.random.uniform(0.95, 1.05)
            rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta),
                                 np.cos(theta), 0], [0, 0, 1]])

            block[:, :3] = np.dot(block_[:, :3], rot_mat) * scale_factor
        else:
            theta = self.angle
            transform_mat = np.array([[np.cos(theta),
                                       np.sin(theta), 0],
                                      [-np.sin(theta),
                                       np.cos(theta), 0], [0, 0, 1]])
            block[...] = block_[...]
            block[:, :3] = np.dot(block[:, :3], transform_mat)

        block[:, 3] = block_[:, 3]
        pc_ = np.round(block[:, :3] / self.voxel_size).astype(np.int32)
        pc_ -= pc_.min(0, keepdims=1)

        if self.ssl_mode and exists(
                self.files[index].replace('lidar',
                                          f"probability_f{self.source}_{self.target}")):
            all_labels = np.load(
                self.files[index].replace('lidar',
                                          f"predictions_f{self.source}_{self.target}")).reshape(-1)

            lcw_ = np.load(self.files[index].replace('lidar',
                                                     f"probability_f{self.source}_{self.target}")).reshape(-1)
            # TODO: check casting
            # lcw_ = (lcw_ * 100).astype(np.int32)
        elif self.ssl_mode:  # in case of GT label give weight = 1.0 per label
            label_file = self.files[index].replace('lidar', 'labels').replace(
                '.npy', '.npy')
            if exists(label_file):
                with open(label_file, 'rb') as a:
                    all_labels = np.load(a)[:, 1].reshape(-1)
            else:
                all_labels = np.zeros(pc_.shape[0]).astype(np.int32)
            lcw_ = np.ones(pc_.shape[0]).astype(np.float32).reshape(-1)
            # TODO: check casting
            # lcw_ = (lcw_ * 100).astype(np.int32)
        else:
            label_file = self.files[index].replace('lidar', 'labels').replace(
                '.npy', '.npy')
            if os.path.exists(label_file):
                with open(label_file, 'rb') as a:
                    all_labels = np.load(a)[:, 1].reshape(-1)
                    # all_labels = np.load(a).reshape(-1)
            else:
                all_labels = np.zeros(pc_.shape[0]).astype(np.int32)

        labels_ = self.label_map[all_labels & 0xFFFF].astype(np.int64)

        feat_ = block

        _, inds, inverse_map = sparse_quantize(pc_,
                                               return_index=True,
                                               return_inverse=True)

        if 'train' in self.split:
            if len(inds) > self.num_points:
                inds = np.random.choice(inds, self.num_points, replace=False)

        # print(f"labels.shape:{labels_.shape}, lcw.shape:{lcw_.shape}")
        pc = pc_[inds]
        feat = feat_[inds]
        labels = labels_[inds]
        lidar = SparseTensor(feat, pc)
        labels = SparseTensor(labels, pc)
        labels_ = SparseTensor(labels_, pc_)
        inverse_map = SparseTensor(inverse_map, pc_)

        if self.ssl_mode and (self.split == 'train'):
            lcw = lcw_[inds]
            lcw = SparseTensor(lcw, pc)
            lcw_ = SparseTensor(lcw_, pc_)

            return {
                'lidar': lidar,
                'targets': labels,
                'lweights': lcw,
                'targets_mapped': labels_,
                'inverse_map': inverse_map,
                'file_name': self.files[index]
            }
        return {
            'lidar': lidar,
            'targets': labels,
            'targets_mapped': labels_,
            'inverse_map': inverse_map,
            'file_name': self.files[index]
        }

    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)
