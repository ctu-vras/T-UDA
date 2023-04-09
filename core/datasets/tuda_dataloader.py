import copy
import glob
import os
import os.path

import numpy as np
import yaml
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
from os.path import exists
from os.path import join

__all__ = ['Universal_TUDA']


class Universal_TUDA(dict):

    def __init__(self, root, voxel_size, num_points, model_configs, modality_hypers, **kwargs):
        submit_to_server = kwargs.get('submit', False)
        sample_stride = kwargs.get('sample_stride', 1)
        google_mode = kwargs.get('google_mode', False)
        ssl_mode = kwargs.get('ssl_mode', False)

        if submit_to_server:
            super().__init__({
                'train':
                    universalTUDALoader(model_configs.data_path.train_data_loader.data_path,
                                        voxel_size,
                                        num_points,
                                        sample_stride=1,
                                        split='train',
                                        submit=True,
                                        model_configs=model_configs),
                'val':
                    universalTUDALoader(model_configs.data_path.val_data_loader.data_path,
                                        voxel_size,
                                        num_points,
                                        sample_stride=1,
                                        split='val',
                                        model_configs=model_configs),
                'test':
                    universalTUDALoader(model_configs.data_path.test_data_loader.data_path,
                                        voxel_size,
                                        num_points,
                                        sample_stride=1,
                                        split='test',
                                        model_configs=model_configs),
                'pseudo':
                    universalTUDALoader(model_configs.data_path.pseudo_data_loader.data_path,
                                        voxel_size,
                                        num_points,
                                        sample_stride=1,
                                        split='pseudo',
                                        model_configs=model_configs)
            })
        else:
            super().__init__({
                'train':
                    universalTUDALoader(model_configs.data_path.train_data_loader.data_path,
                                        voxel_size,
                                        num_points,
                                        sample_stride=1,
                                        split='train',
                                        model_configs=model_configs,
                                        modality_hypers=modality_hypers,
                                        google_mode=google_mode,
                                        ssl_mode=ssl_mode),
                'val':
                    universalTUDALoader(model_configs.data_path.val_data_loader.data_path,
                                        voxel_size,
                                        num_points,
                                        sample_stride=sample_stride,
                                        split='val',
                                        model_configs=model_configs,
                                        modality_hypers=modality_hypers),
                'test':
                    universalTUDALoader(model_configs.data_path.test_data_loader.data_path,
                                        voxel_size,
                                        num_points,
                                        sample_stride=sample_stride,
                                        split='test',
                                        model_configs=model_configs,
                                        modality_hypers=modality_hypers),
                'pseudo':
                    universalTUDALoader(model_configs.data_path.ssl_data_loader.data_path,
                                        voxel_size,
                                        num_points,
                                        sample_stride=sample_stride,
                                        split='pseudo',
                                        model_configs=model_configs,
                                        modality_hypers=modality_hypers)
            })


def transform_pcl_scan(points, pose0, pose):
    # pose = poses[0][idx]

    hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
    # new_points = hpoints.dot(pose.T)
    new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)

    new_points = new_points[:, :3]
    new_coords = new_points - pose0[:3, 3]
    # new_coords = new_coords.dot(pose0[:3, :3])
    new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
    new_coords = np.hstack((new_coords, points[:, 3:]))

    return new_coords


def fuse_multiscan(ref_raw_data, ref_annotated_data, ref_lcw, transformed_data,
                   transformed_annotated_data, transformed_lcw, source, ssl_mode):
    lcw = None
    if (source != 1) and (source != -1):
        print(f"Error data source {source} not Implemented")
        return 0
    if source == -1:  # past frame
        raw_data = np.concatenate((transformed_data, ref_raw_data), 0)
        annotated_data = np.concatenate((transformed_annotated_data, ref_annotated_data), 0)
        if ssl_mode:
            lcw = np.concatenate((transformed_lcw, ref_lcw), 0)

    if source == 1:  # future frame
        raw_data = np.concatenate((ref_raw_data, transformed_data,), 0)
        annotated_data = np.concatenate((ref_annotated_data, transformed_annotated_data), 0)
        if ssl_mode:
            lcw = np.concatenate((ref_lcw, transformed_lcw), 0)

    return raw_data, annotated_data, lcw


class universalTUDALoader:

    def __init__(self,
                 root,
                 voxel_size,
                 num_points,
                 split,
                 model_configs,
                 modality_hypers,
                 sample_stride=1,
                 submit=False,
                 google_mode=False,
                 ssl_mode=False):
        if submit:
            trainval = True
        else:
            trainval = False
        modality_hypers = modality_hypers
        mapping = model_configs.data.label_mapping
        with open(mapping, 'r') as stream:
            mapping_config = yaml.safe_load(stream)

        self.learning_map = mapping_config['learning_map']
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.sample_stride = sample_stride
        self.google_mode = google_mode
        self.ssl_mode = ssl_mode
        self.source = modality_hypers["source"]
        self.target = modality_hypers["target"]
        self.past = modality_hypers["past"]
        self.future = modality_hypers["future"]
        self.rgb = modality_hypers["rgb"]
        self.UDA = modality_hypers["uda"]
        self.use_time = modality_hypers["time"]
        # self.use_intensity = modality_hypers["intensity"]

        self.seqs = []

        if split == 'train':
            self.seqs = mapping_config['split']['train']
            self.sensor_zpose = modality_hypers["S_sensor_zpose"]
            if self.google_mode or trainval:
                self.seqs.append('08')
        elif split == 'val':
            self.seqs = mapping_config['split']['valid']
            self.sensor_zpose = modality_hypers["T_sensor_zpose"]
        elif split == 'test':
            self.seqs = mapping_config['split']['test']
            self.sensor_zpose = modality_hypers["T_sensor_zpose"]
        elif split == 'pseudo':
            self.seqs = mapping_config['split']['pseudo']
            self.sensor_zpose = modality_hypers["T_sensor_zpose"]
        else:
            raise Exception(f'{split}: Split must be train/val/test/pseudo')

        if os.path.exists(os.path.join(self.root, self.seqs[0], 'velodyne')):
            self.get_data = self.get_bin_data
            pcl_folder = 'velodyne'
            self.load_poses = self.load_kitti_calib_poses

        elif os.path.exists(os.path.join(self.root, self.seqs[0], 'lidar')):
            self.get_data = self.get_npy_data
            pcl_folder = 'lidar'
            self.load_poses = self.load_wod_poses
        else:
            raise Exception("point cloud folder is neither 'velodyne' nor 'lidar'...")

        self.files = []
        for seq in self.seqs:
            seq_file = sorted(
                os.listdir(os.path.join(self.root, seq, pcl_folder)))
            seq_files = [
                os.path.join(self.root, seq, pcl_folder, x) for x in seq_file
            ]
            self.files.extend(seq_files)

        if self.sample_stride > 1:
            self.files = self.files[::self.sample_stride]

        self.reverse_label_name_mapping = mapping_config['learning_map_inv']
        self.label_map = self.learning_map
        self.num_classes = model_configs.data.num_classes
        self.angle = 0.0

        if self.past or self.future:
            self.load_poses()

    def set_angle(self, angle):
        self.angle = angle

    #############################################################
    # load kitti type of poses
    def load_kitti_calib_poses(self):
        """
        load calib poses and times.
        """

        ###########
        # Load data
        ###########

        self.calibrations = []
        # self.times = []
        self.poses = {}  # []

        for seq in self.seqs:
            seq_folder = join(self.root, str(seq))

            # Read Calib
            self.calibrations.append(self.parse_calibration(join(seq_folder, "calib.txt")))

            # Read times
            # self.times.append(np.loadtxt(join(seq_folder, 'times.txt'), dtype=np.float32))

            # Read poses
            poses_f64 = self.parse_poses(join(seq_folder, 'poses.txt'), self.calibrations[-1])
            # self.poses.append([pose.astype(np.float32) for pose in poses_f64])
            self.poses[seq] = [pose.astype(np.float32) for pose in poses_f64]

    def parse_calibration(self, filename):
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

    def parse_poses(self, filename, calibration):
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

    # load wod type of poses
    def load_wod_poses(self):
        """
        load calib poses and times.
        """
        ###########
        # Load data
        ###########

        self.calibrations = []
        # self.times = []
        self.poses = {}  # []

        for k, seq in enumerate(self.seqs):  # range(0, 22):
            seq_folder = join(self.root, str(seq))

            # Read poses
            poses_f64 = self.parse_poses(seq_folder, k)
            # self.poses.append([pose.astype(np.float32) for pose in poses_f64])
            self.poses[seq] = [pose.astype(np.float32) for pose in poses_f64]

    def parse_poses(self, seq, k):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        filename = sorted(glob.glob(os.path.join(seq, "poses", "*.npy")))

        poses = []

        for file in filename:
            pose = np.load(file)
            poses.append(pose)
        return poses

    # parse .bin format point clouds (e.g. kitti original data format)
    def get_bin_data(self, newpath, time_frame_idx):
        raw_data = np.fromfile(newpath, dtype=np.float32).reshape((-1, 4))
        lcw = None
        if self.use_time:
            raw_data[:, 3] = np.ones_like(raw_data[:, 3]) * time_frame_idx

        if self.UDA:
            raw_data[:, 2] += self.sensor_zpose  # elevate the point cloud two meters up to align with WOD

        if self.split == 'pseudo':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            if self.ssl_mode and exists(newpath.replace('velodyne', f"predictions_f{self.source}_{self.target}")[:-3]
                                        + 'label'):
                annotated_data = np.fromfile(
                    newpath.replace('velodyne', f"predictions_f{self.source}_{self.target}")[:-3] + 'label',
                    dtype=np.int32).reshape(-1)
            else:
                annotated_data = np.fromfile(newpath.replace('velodyne', 'mapped_labels')[:-3] + 'label',
                                             dtype=np.int32).reshape(-1)
            # if np.sum(np.unique(annotated_data == 18)) > 0:
            #     print(newpath)

            if self.ssl_mode and exists(newpath.replace('velodyne', f"probability_f{self.source}_{self.target}")[
                                        :-3] + 'label'):
                lcw = np.fromfile(
                    newpath.replace('velodyne', f"probability_f{self.source}_{self.target}")[
                    :-3] + 'label',
                    dtype=np.float32).reshape(-1)
                # TODO: check casting
                # lcw = (lcw * 100).astype(np.float32)

            elif self.ssl_mode:  # in case of GT label give weight = 1.0 per label
                lcw = np.expand_dims(np.ones_like(raw_data[:, 0], dtype=np.float32), axis=1)
                # TODO: check casting
                # lcw = (lcw * 100).astype(np.float32)
        # only get the x y z i/t
        raw_data = raw_data[:, :4]
        return raw_data, annotated_data.astype(np.int32), len(raw_data), lcw

    def get_npy_data(self, newpath, time_frame_idx):
        raw_data = np.load(newpath)
        if raw_data.shape[1] == 3:
            data = np.zeros((len(raw_data), 4))
            data[:, :3] = raw_data[:, :3]
            raw_data = copy.copy(data)
        # print(newpath, raw_data.shape)
        # assert raw_data.shape[1] >= 4

        # if self.use_intensity:
        #     data = np.zeros((len(raw_data), 5))  # initialize 5 feature input data
        #     data[:, :3] = raw_data[:, :3]
        #     data[:, 4] = raw_data[:, 3]  # add intensity as 5th feature
        #     raw_data = copy.copy(data)

        if self.use_time:
            # print(self.use_time)
            raw_data[:, 3] = np.ones_like(raw_data[:, 3]) * time_frame_idx
        if self.UDA:
            raw_data[:, 2] += self.sensor_zpose  # elevate the point cloud two meters up to align with WOD

        # # TODO: check if the colors are encoded correctly instead of the lidar intensity
        # if self.rgb:
        #     # load rgb colors for each points
        #     raw_rgb = np.load(newpath.replace('lidar', 'colors')[
        #                       :-3] + 'npy')
        #     # convert rgb into gray scale [0, 255]
        #     raw_gray = 0.2989 * raw_rgb[:, 0] + 0.5870 * raw_rgb[:, 1] + 0.1140 * raw_rgb[:, 2]
        #     # mask (0) ignored point colors  (originally not provided on wod rear-cameras) -> rgb:[1,1,1] or gray:[
        #     # 0.99990])
        #     gray_mask = raw_gray > 1  # < 1 #0.9998999999999999
        #     # assign 0 to the place we want to mask
        #     raw_gray[gray_mask] = -1
        #     # replace intensity with gray scale camera image/frame color
        #     raw_data[:, 3] = raw_gray
        #     # raw_data[:,4] = gray_mask * 1
        lcw = None
        origin_len = len(raw_data)
        if self.split == 'pseudo':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0]), axis=1).reshape(-1).astype(np.int32)
        else:
            # x = self.im_idx[index].replace('lidar', f"predictions_f{self.source}_{self.target}")[:-3] + 'label'
            if self.ssl_mode and exists(newpath.replace('lidar', f"predictions_f{self.source}_{self.target}")[
                                        :-3] + 'npy'):
                annotated_data = np.load(
                    newpath.replace('lidar', f"predictions_f{self.source}_{self.target}")[
                    :-3] + 'npy').reshape(-1).astype(np.int32)
            else:
                # print(self.im_idx[index].replace('lidar', 'mapped_labels')[:-3] + 'npy')
                annotated_data = np.load(newpath.replace('lidar', 'mapped_labels')[:-3] + 'npy',
                                         allow_pickle=True)
                if len(annotated_data.shape) == 2:
                    if annotated_data.shape[1] == 2:
                        annotated_data = annotated_data[:, 1]
                # Reshape the label/annotation to vector.
                annotated_data = annotated_data.reshape(-1).astype(np.int32)

            # annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary

            if self.ssl_mode and exists(newpath.replace('lidar', f"probability_f{self.source}_{self.target}")[
                                        :-3] + 'npy'):
                lcw = np.load(newpath.replace('lidar', f"probability_f{self.source}_{self.target}")[
                              :-3] + 'npy').reshape(-1).astype(np.float32)
                # TODO: check casting
                # lcw = (lcw * 100).astype(np.int32)
            elif self.ssl_mode:  # in case of GT label give weight = 1.0 per label
                lcw = np.expand_dims(np.ones_like(raw_data[:, 0]), axis=1).astype(np.float32)
                # TODO: check casting
                # lcw = (lcw * 100).astype(np.int32)
        # only get the x y z i/t
        # raw_data = raw_data[:, :4]

        return raw_data.astype(np.float32), annotated_data, len(raw_data), lcw

    #####################################################################

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # -----------------------------------------------------------------Begin
        # reference scan
        reference_file = self.files[index]
        raw_data, annotated_data, data_len, lcw = self.get_data(reference_file, 0)

        origin_len = data_len

        number_idx = int(self.files[index][-10:-4])
        # dir_idx = int(self.files[index][-22:-20])
        dir_idx = self.files[index].split('/')[-3]

        # past scan
        past_frame_len = 0
        # TODO: added the future frame availability check
        if self.past and ((number_idx - self.past) >= 0) and ((number_idx + self.past) < len(self.poses[dir_idx])):
            # extract the poss of the reference frame
            pose0 = self.poses[dir_idx][number_idx]
            for fuse_idx in range(self.past):
                # TODO: past frames
                frame_ind = fuse_idx + 1
                pose = self.poses[dir_idx][number_idx - frame_ind]
                past_file = self.files[index][:-10] + str(number_idx - frame_ind).zfill(6) + self.files[index][-4:]
                past_raw_data, past_annotated_data, past_data_len, past_lcw = self.get_data(past_file,
                                                                                            -frame_ind)

                past_raw_data = transform_pcl_scan(past_raw_data, pose0, pose)

                # past frames
                if past_data_len != 0:
                    raw_data, annotated_data, lcw = fuse_multiscan(raw_data, annotated_data, lcw,
                                                                   past_raw_data, past_annotated_data, past_lcw, -1,
                                                                   self.ssl_mode)
                    # count number of past frame points
                    past_frame_len += past_data_len

        # future scan
        future_frame_len = 0
        # TODO: added the future frame availability check
        if self.future and ((number_idx - self.future) >= 0) and (
                (number_idx + self.future) < len(self.poses[dir_idx])):
            # extract the poss of the reference frame
            pose0 = self.poses[dir_idx][number_idx]
            for fuse_idx in range(self.future):
                # TODO: future frame
                frame_ind = fuse_idx + 1
                future_pose = self.poses[dir_idx][number_idx + frame_ind]
                future_file = self.files[index][:-10] + str(number_idx + frame_ind).zfill(6) + self.files[index][-4:]
                future_raw_data, future_annotated_data, future_data_len, future_lcw = self.get_data(
                    future_file, frame_ind)

                future_raw_data = transform_pcl_scan(future_raw_data, pose0, future_pose)

                # TODO: check correctness (future frame)
                if future_data_len != 0:
                    raw_data, annotated_data, lcw = fuse_multiscan(raw_data, annotated_data, lcw,
                                                                   future_raw_data, future_annotated_data, future_lcw,
                                                                   1, self.ssl_mode)
                    # count number of future frame points
                    future_frame_len += future_data_len
        # reference frame index position [start, end] where end: start + length of reference
        reference_idx = [past_frame_len, past_frame_len + origin_len]
        # -----------------------------------------------------------------End

        block_, all_labels, lcw_ = raw_data, annotated_data, lcw

        # with open(self.files[index], 'rb') as b:
        #     block_ = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
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

        all_labels = (all_labels & 0xFFFF).astype(np.uint32)
        # print(all_labels)

        labels_ = np.vectorize(self.label_map.__getitem__)(all_labels)
        # print(labels_.shape)
        feat_ = block

        _, inds, inverse_map = sparse_quantize(pc_,
                                               return_index=True,
                                               return_inverse=True)

        ref_pc_ = pc_[reference_idx[0]: reference_idx[1]]
        ref_labels_ = labels_[reference_idx[0]: reference_idx[1]]
        ref_feat_ = feat_[reference_idx[0]: reference_idx[1]]

        _, ref_inds, ref_inverse_map = sparse_quantize(ref_pc_,
                                                       return_index=True,
                                                       return_inverse=True)

        # if 'train' in self.split:
        #     if len(inds) > self.num_points:
        #         inds = np.random.choice(inds, self.num_points, replace=False)

        # print(f"labels.shape:{labels_.shape}, lcw.shape:{lcw_.shape}")
        pc = pc_[inds]
        feat = feat_[inds]
        labels = labels_[inds]
        sparse_reference_idx = (inds >= reference_idx[0]) & (inds <= reference_idx[1])

        ref_pc = ref_pc_[ref_inds]
        ref_feat = ref_feat_[ref_inds]
        ref_labels = ref_labels_[ref_inds]

        # ref_all_pc = pc[sparse_reference_idx]


        ref_lidar = SparseTensor(ref_feat, ref_pc)
        ref_labels = SparseTensor(ref_labels, ref_pc)
        ref_labels_ = SparseTensor(ref_labels_, ref_pc_)
        ref_inverse_map = SparseTensor(ref_inverse_map, ref_pc_)

        lidar = SparseTensor(feat, pc)
        labels = SparseTensor(labels, pc)
        labels_ = SparseTensor(labels_, pc_)
        inverse_map = SparseTensor(inverse_map, pc_)
        # for calculating loss
        reference_idx = SparseTensor(sparse_reference_idx, pc)


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
                'file_name': self.files[index],
                'reference_idx': sparse_reference_idx
            }
        return {
            'lidar': lidar,
            'targets': labels,
            'targets_mapped': labels_,
            'inverse_map': inverse_map,
            'file_name': self.files[index],
            'reference_idx': reference_idx,
            'ref_lidar': ref_lidar,
            'ref_targets': ref_labels,
            'ref_targets_mapped': ref_labels_,
            'ref_inverse_map': ref_inverse_map
        }

    @staticmethod
    def collate_fn(inputs):
        return sparse_collate_fn(inputs)


# load label class info
def get_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        config_yaml = yaml.safe_load(stream)
    class_label_name = dict()
    for i in sorted(list(config_yaml['learning_map'].keys()))[::-1]:
        class_label_name[config_yaml['learning_map'][i]] = config_yaml['labels'][i]

    return class_label_name
