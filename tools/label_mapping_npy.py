# -*- coding:utf-8 -*-
# author: Awet H. Gebrehiwot
# at 10/13/22
# --------------------------|
import numpy as np
import glob
import os
from os.path import join

semanticKitti_learning_map = {
    0: 0,  # "unlabeled"
    1: 0,  # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 1,  # "car"
    11: 2,  # "bicycle"
    13: 4,  # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 3,  # "motorcycle"
    16: 4,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 6,  # "truck"
    20: 4,  # "other-vehicle"
    30: 5,  # "person"
    31: 2,  # "bicyclist"
    32: 3,  # "motorcyclist"
    40: 7,  # "road"
    44: 7,  # "parking"
    48: 8,  # "sidewalk"
    49: 10,  # "other-ground"
    50: 11,  # "building"
    51: 11,  # "fence"
    52: 11,  # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 7,  # "lane-marking" to "road" ---------------------------------mapped
    70: 10,  # "vegetation"
    71: 10,  # "trunk"
    72: 9,  # "terrain"
    80: 11,  # "pole"
    81: 11,  # "traffic-sign"
    99: 11,  # "other-object" to "unlabeled" ----------------------------mapped
    252: 1,  # "moving-car" to "car" ------------------------------------mapped
    253: 2,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 5,  # "moving-person" to "person" ------------------------------mapped
    255: 3,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 4,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 4,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 6,  # "moving-truck" to "truck" --------------------------------mapped
    259: 4  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}

nusceneces_learning_map = {
    1: 0,  # 'noise'
    5: 0,
    7: 0,
    8: 0,
    10: 0,
    11: 0,
    13: 0,
    19: 0,
    20: 0,
    0: 0,
    29: 0,
    31: 0,
    17: 1,  # 'car'
    14: 2,  # 'bicycle'
    21: 3,  # 'motorcycle'
    15: 4,  # 'other-vehicle'
    16: 4,
    18: 4,
    2: 5,  # 'pedestrian'
    3: 5,
    4: 5,
    6: 5,
    23: 6,  # 'truck'
    24: 7,  # drivable-surface
    26: 8,  # sidewalk
    27: 9,  # 'terrain'
    30: 10,  # 'vegetation'
    9: 11,  # manmade
    12: 11,
    22: 11,
    25: 11,  #
    28: 11
}

wod_learning_map = {
    0: 0,  # "Undefined"
    1: 1,  # "Car"
    2: 6,  # "Truck"
    3: 2,  # "Bus"
    4: 4,  # "Other Vehicle" # Other small vehicles (e.g. pedicab), large vehicles (e.g. const vehicl, RV, limo, tram).
    5: 3,  # "Motorcyclist"
    6: 2,  # "Bicyclist"
    7: 5,  # "Pedestrian"
    8: 11,  # "Sign"
    9: 11,  # "Traffic Light"
    10: 11,  # "Pole" # Lamp post, traffic sign pole etc.
    11: 11,  # "Construction Cone" # Construction cone/pole.
    12: 2,  # "Bicycle"
    13: 3,  # "Motorcycle"
    14: 11,  # "Building"
    15: 10,  # "Vegetation" # Bushes, tree branches, tall grasses, flowers etc.
    16: 11,  # "Tree Trunk"
    17: 8,  # "Curb" # Curb on the edge of roads. This does not include road boundaries if there’s no curb.
    18: 7,
    # "Road" # Surface a vehicle could drive on. This include the driveway connecting // parking lot and road over a section of sidewalk.
    19: 7,
    # "Lane Marker" # Marking on the road that’s specifically for defining lanes such as // single/double white/yellow lines.
    20: 7,  # "Other Ground" # Marking on the road other than lane markers, bumps, cateyes, railtracks // etc.
    21: 9,
    # "Walkable" # Most horizontal surface that’s not drivable, e.g. grassy hill, // pedestrian walkway stairs etc.
    22: 8  # "Sidewalk" # Nicely paved walkable surface when pedestrians most likely to walk on.
}


def convert_labels(data_path, dest_path, dataset):
    sequeses = os.listdir(data_path)
    if dataset == 'WOD':
        label_mapping = wod_learning_map
    elif dataset == 'SemanticKITTI':
        label_mapping = semanticKitti_learning_map
    elif dataset == 'NuSceneces':
        label_mapping = nusceneces_learning_map
    for seq in sequeses:
        frames = sorted(glob.glob(os.path.join(data_path, seq, 'labels', '*.npy')))
        seq_folder = join(data_path, seq)

        label_path = os.path.join(dest_path, seq, 'mapped_labels')
        if not os.path.exists(label_path):
            os.makedirs(label_path)

        for frame in frames:
            ss = np.load(frame) & 0xFFFF
            mapped_labels = (np.vectorize(label_mapping.__getitem__)(ss)).astype(np.int32)
            new_frame = frame.split('/')[-1].split('.')[0]
            np.save(os.path.join(label_path, new_frame), mapped_labels)


if __name__ == '__main__':
    # data_path = '/mnt/personal/gebreawe/Datasets/RealWorld/semantic-kitti/all_npy/sequences'
    # dest_path = '/mnt/personal/gebreawe/Datasets/RealWorld/semantic-kitti/all_npy/sequences'
    # dataset = 'SemanticKITTI'

    # data_path = '/mnt/personal/gebreawe/Datasets/RealWorld/WOD/processed/Labeled_64_beam/training'
    # dest_path = '/mnt/personal/gebreawe/Datasets/RealWorld/WOD/processed/Labeled_64_beam/training'
    # dataset = 'WOD'
    # data_path = '/mnt/personal/gebreawe/Datasets/RealWorld/WOD/processed/Labeled_32_beam/training'
    # dest_path = '/mnt/personal/gebreawe/Datasets/RealWorld/WOD/processed/Labeled_32_beam/training'
    #
    # data_path = '/mnt/personal/gebreawe/Datasets/RealWorld/WOD/processed/Labeled_64_beam/validation'
    # dest_path = '/mnt/personal/gebreawe/Datasets/RealWorld/WOD/processed/Labeled_64_beam/validation'
    #
    # data_path = '/mnt/personal/gebreawe/Datasets/RealWorld/WOD/processed/Labeled_32_beam/validation'
    # dest_path = '/mnt/personal/gebreawe/Datasets/RealWorld/WOD/processed/Labeled_32_beam/validation'

    # data_path = '/mnt/personal/gebreawe/Datasets/RealWorld/NUSCENES/processed/train'
    # dest_path = '/mnt/personal/gebreawe/Datasets/RealWorld/NUSCENES/processed/train'
    dataset = 'NuSceneces'
    dest_path = '/mnt/personal/gebreawe/Datasets/RealWorld/NUSCENES/processed/val'

    # 'NuSceneces'
    convert_labels(data_path, dest_path, dataset)
