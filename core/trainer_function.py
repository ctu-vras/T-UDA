# -*- coding:utf-8 -*-
# author: Awet
# date: 1/11/2022

import copy
from typing import Any, Callable, Dict

import numpy as np
import torch
import torchsparse

from torchsparse import SparseTensor

from torch import nn
from torch.cuda import amp
from torchpack.train import Trainer
from torchpack.utils.typing import Optimizer, Scheduler
from core.metric_util import per_class_iu, fast_hist_crop
from tqdm import tqdm

__all__ = ['TUDATrainer']


def yield_target_dataset_loader(n_epochs, target_train_dataset_loader):
    for e in range(n_epochs):
        for data_feed in target_train_dataset_loader:
            # print(data_feed)
            yield data_feed


def get_devoxelized_outputs(inputs, outputs, feed_dict, mode='T'):
    if mode == 'T':
        invs = feed_dict['inverse_map']
        all_labels = feed_dict['targets_mapped']
    elif mode == 'S':
        invs = feed_dict['ref_inverse_map']
        all_labels = feed_dict['ref_targets_mapped']
    _outputs = []
    _targets = []
    for idx in range(invs.C[:, -1].max() + 1):
        cur_scene_pts = (inputs.C[:, -1] == idx).cpu().numpy()
        cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
        cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
        outputs_mapped = outputs[cur_scene_pts][cur_inv].argmax(1)
        targets_mapped = all_labels.F[cur_label]
        _outputs.append(outputs_mapped)
        _targets.append(targets_mapped)
    outputs = torch.cat(_outputs).cpu().numpy()
    targets = torch.cat(_targets).cpu().numpy()
    # print(f"mode: {mode}, outputs:{outputs.shape}, targets{targets.shape}")
    return outputs, targets


# class TUDATrainer(Trainer):
class TUDATrainer:

    def __init__(self,
                 model_student: nn.Module,
                 model_teacher: nn.Module,
                 criterion: Callable,
                 optimizer_teacher: Optimizer,
                 optimizer_student: Optimizer,
                 scheduler_student: Scheduler,
                 scheduler_teacher: Scheduler,
                 num_workers: int,
                 warmup_epoch: int,
                 max_num_epochs: int,
                 target_dataset_loader: Dict[str, Any],
                 num_classes: int,
                 ignore_label: int,
                 unique_label: int,
                 unique_label_str: str,
                 run_dir: dir(),
                 seed: int,
                 amp_enabled: bool = False,
                 ssl_mode: bool = False) -> None:
        self.student_model = model_student
        self.teacher_model = model_teacher
        self.start_uda = False
        self.criterion = criterion
        self.optimizer_teacher = optimizer_teacher
        self.optimizer_student = optimizer_student
        self.scheduler_student = scheduler_student
        self.scheduler_teacher = scheduler_teacher
        self.num_workers = num_workers
        self.warmup_epoch = warmup_epoch
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.unique_label = unique_label
        self.unique_label_str = unique_label_str
        self.ema_update_now = False
        self.gama = 1.0  # weight of self-supervised loss
        self.conf_thr = 0.0  # confidence threshold
        self.seed = seed
        self.amp_enabled = amp_enabled
        self.scaler = amp.GradScaler(enabled=self.amp_enabled)
        self.epoch_num = 1
        self.max_num_epochs = max_num_epochs
        self.ssl_mode = ssl_mode
        self.one_epoch_tracker = 0
        self.val_teacher = False
        self.val_student = False
        self.teacher_best_val_miou = 0
        self.student_best_val_miou = 0
        self.ema_update_now = False
        self.teacher_model_save_path = run_dir + f"{self.conf_thr}_best_teacher.pt"
        self.student_model_save_path = run_dir + f"{self.conf_thr}_best_student.pt"
        self.target_data_generator = yield_target_dataset_loader(self.max_num_epochs, target_dataset_loader)

    def calculate_loss(self, outputs, targets, _lcw=None):
        if self.start_uda:
            # print(outputs.size(), targets.size())
            raw_loss = self.criterion(outputs, targets)
            # print(f"raw_loss:{raw_loss.size()}, lcw: {lcw.F}")
            # extract the values/features from th SpraseTensor using ".F"
            mask = _lcw >= self.conf_thr
            lcw = mask * _lcw
            w_loss = raw_loss * torch.squeeze(lcw)
            loss = w_loss.mean()  # / 100.0
            print(f"_lcw:{_lcw}, lcw: {lcw}")
        elif self.ssl_mode:
            # print(outputs.size(), targets.size())
            raw_loss = self.criterion(outputs, targets)

            lcw = _lcw.F
            # print(f"raw_loss:{raw_loss.size()}, lcw: {lcw.F}")
            # extract the values/features from th SpraseTensor using ".F"
            mask = lcw >= self.conf_thr
            lcw = mask * lcw
            w_loss = raw_loss * torch.squeeze(lcw)
            loss = w_loss.mean() / 100.0
            # print(f"w_loss:{w_loss.size()}, loss: {loss}")
        else:
            # print(outputs.size(), targets.size())
            loss = self.criterion(outputs, targets)

            # print(f"loss.size:{loss.size()}, loss: {loss.item()}")
        return loss

    # initialize student model from teacher model weights
    @torch.no_grad()
    def _initialize_student_model_from_teacher(self):
        teacher_model_dict = self.teacher_model.state_dict()
        self.student_model.load_state_dict(teacher_model_dict)

    # updating teacher model weights
    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.996):
        student_model_dict = self.student_model.state_dict()

        new_teacher_dict = copy.copy(self.teacher_model.state_dict())  # OrderedDict()
        # print(f"before: {self.teacher_model.state_dict()}")
        for key, value in self.teacher_model.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                        student_model_dict[key] *
                        (1 - keep_rate) + value * keep_rate
                )
            # else:
            #    print("{} is not found in student model".format(key))
            #    #raise Exception("{} is not found in student model".format(key))
        # print(f"new_teacher_dict: {new_teacher_dict}")
        self.teacher_model.load_state_dict(new_teacher_dict)

    def validate_uda(self, val_dataset_loader, mode='T'):
        teacher_hist_list = []
        student_hist_list = []
        with torch.no_grad():
            for i_iter_val, feed_dict in enumerate(val_dataset_loader):
                _inputs = {}
                for key, value in feed_dict.items():
                    if ('name' not in key) and ('reference_idx' not in key):
                        _inputs[key] = value.cuda()

                if self.val_teacher:
                    self.teacher_model.eval()
                    inputs = _inputs['lidar']
                    outputs = self.teacher_model(inputs)
                    # TODO: check if this is correctly implemented
                    outputs_teacher, targets_teacher = get_devoxelized_outputs(inputs, outputs, feed_dict, mode='T')
                    # print("teacher", outputs_teacher.shape, targets_teacher.shape)
                if self.val_student:
                    self.student_model.eval()
                    inputs = _inputs['ref_lidar']
                    outputs = self.student_model(inputs)
                    outputs_student, targets_student = get_devoxelized_outputs(inputs, outputs, feed_dict, mode='S')
                    # print("student", outputs_student.shape, targets_student.shape)
                    # np.save('_'.join(feed_dict['file_name'][0].split('/')[-3:]),  outputs_student)
                if self.val_teacher:
                    teacher_hist_list.append(fast_hist_crop(outputs_teacher, targets_teacher, self.unique_label))

                if self.val_student:
                    student_hist_list.append(fast_hist_crop(outputs_student, targets_student, self.unique_label))

        return teacher_hist_list, student_hist_list

    def forward(self, model, inputs, mode='Train'):
        grad = False
        if mode == 'Train':
            model.train()
            grad = True
        elif mode == 'Pseudo_Labeling':
            model.eval()
            grad = False
        with torch.set_grad_enabled(grad):
            # forward + backward + optimize
            outputs = model(inputs)
            # print(f"outputs.size() : {outputs.size()}")
            return outputs

    def uda_fit(self, source_train_dataset_loader, target_train_dataset_loader, val_dataset_loader,
                test_loader=None, ckpt_save_interval=5, lr_scheduler_each_iter=False):
        self.target_data_generator = yield_target_dataset_loader(self.max_num_epochs, target_train_dataset_loader)
        data_length = len(source_train_dataset_loader)
        pbar_interval = int(data_length / 1000 + 1)
        global_iter = 1
        for epoch in range(self.max_num_epochs):
            pbar = tqdm(total=data_length)
            # train the model
            loss_list = []
            # switch the teacher model validation to False
            self.val_teacher = False
            # switch the student model validation to False
            self.val_student = False
            # switch teacher weight update using ema to False
            self.ema_update_now = False
            # training with multi-frames and ssl:
            for i_iter_train, feed_dict in enumerate(source_train_dataset_loader):
                _inputs = {}
                for key, value in feed_dict.items():
                    if ('name' not in key) and ('reference_idx' not in key):
                        _inputs[key] = value.cuda()

                inputs_source = _inputs['lidar']
                labels_source = feed_dict['targets'].F.long().cuda(non_blocking=True)
                # print(inputs_source.F.size(), labels_source.size())
                # print(feed_dict['file_name'])

                # print(f"self.epoch_num: {self.epoch_num}, self.local_step {self.local_step}")
                if (epoch == self.warmup_epoch) and (i_iter_train == 0):
                    # initialize a student model form teacher model weights
                    self._initialize_student_model_from_teacher()
                    # student_model_dict = self.student_model.state_dict()
                    # teacher_model_dict = self.teacher_model.state_dict()
                    # print(f"1student_model_dict: {student_model_dict['stem.0.kernel']}")
                    # print(f"1teacher_model_dict: {teacher_model_dict['stem.0.kernel']}")

                #####################################################
                # train teacher model in the burn in stage
                if epoch < self.warmup_epoch:
                    self.teacher_model.train()
                    source_output = self.forward(self.teacher_model, inputs_source, mode='Train')

                    loss = self.criterion(source_output, labels_source)

                    # TODO: check --> to mitigate only one element tensors can be converted to Python scalars
                    self.optimizer_teacher.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer_teacher)
                    self.scaler.update()
                    self.scheduler_teacher.step()

                    loss_list.append(loss.item())
                    # switch the teacher model validation to True
                    self.val_teacher = True

                ################################
                # T-UDA: Student - Teacher mutual learning
                else:
                    # ----------------------------------------------------#
                    # Student <-------> Teacher mutual learning block
                    # Change teacher model to evaluation mode
                    # student_model_dict = self.student_model.state_dict()
                    # teacher_model_dict = self.teacher_model.state_dict()
                    # print(f"b eval student_model_dict: {student_model_dict['stem.0.kernel']}")
                    # print(f"b eval teacher_model_dict: {teacher_model_dict['stem.0.kernel']}")
                    self.teacher_model.eval()
                    # change student model to training mode
                    self.student_model.train()
                    # Student forward pass on Source data
                    source_sample_feat = inputs_source.F
                    source_sample_chanel = inputs_source.C
                    reference_frame_idx = feed_dict['reference_idx']
                    # print(f"reference_frame_idx: {reference_frame_idx.F.size()}")
                    source_ref_sample = SparseTensor(source_sample_feat[reference_frame_idx.F],
                                                     source_sample_chanel[reference_frame_idx.F])
                    ref_labels_source = labels_source[reference_frame_idx.F]
                    # student forward pass
                    source_output = self.forward(self.student_model, source_ref_sample, mode='Train')

                    # --- Pseudo Labeling--------#
                    # load target data
                    feed_dict_target_data = next(self.target_data_generator)
                    _inputs_targets = {}
                    for key, value in feed_dict_target_data.items():
                        if ('name' not in key) and ('reference_idx' not in key):
                            _inputs_targets[key] = value.cuda()
                    inputs_target_data = _inputs_targets['lidar']
                    # targets_target_data = feed_dict_target_data['targets'].F.long().cuda(non_blocking=True)
                    # inference/pseudo labeling of target data
                    target_prediction = self.forward(self.teacher_model, inputs_target_data, mode='Pseudo_Labeling')
                    # pseudo label generated by teacher model
                    pseudo_label = torch.squeeze(torch.argmax(target_prediction, dim=1))
                    # calculate weight/confidence of each pseudo label
                    predict_probability = torch.nn.functional.softmax(target_prediction, dim=1)
                    # pseudo label confidence ---> lcw
                    pseudo_labels_prob_lcw, predict_prob_ind = predict_probability.max(dim=1)
                    # multiply by 100
                    pseudo_labels_prob_lcw = pseudo_labels_prob_lcw
                    # Create a lcw tensor of ones for the source data and multiply by 100
                    source_lcw = torch.ones_like(ref_labels_source)
                    ###
                    reference_frame_idx = feed_dict_target_data['reference_idx']
                    target_sample_feat = inputs_target_data.F
                    target_sample_chanel = inputs_target_data.C
                    # print(f"reference_frame_idx: {reference_frame_idx.F.size()}")
                    target_ref_sample = SparseTensor(target_sample_feat[reference_frame_idx.F],
                                                     target_sample_chanel[reference_frame_idx.F])
                    target_ref_pseudo_label = pseudo_label[reference_frame_idx.F]
                    target_ref_pseudo_labels_prob_lcw = pseudo_labels_prob_lcw[reference_frame_idx.F]

                    # Student forward pass on target data
                    outputs_target_data = self.forward(self.student_model, target_ref_sample, mode='Train')
                    # print(f"inpust: {inputs.F}")

                    loss_source_data = self.calculate_loss(source_output, ref_labels_source, source_lcw)
                    loss_target_data = self.calculate_loss(outputs_target_data, target_ref_pseudo_label,
                                                           target_ref_pseudo_labels_prob_lcw)
                    # print(loss_source_data, loss_target_data)

                    loss = loss_source_data + self.gama * loss_target_data

                    # TODO: check --> to mitigate only one element tensors can be converted to Python scalars
                    # loss = loss.mean()
                    # print(loss)
                    # loss.backward()
                    # self.optimizer_student.step()
                    # self.optimizer_student.zero_grad()
                    self.optimizer_student.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer_student)
                    self.scaler.update()
                    self.scheduler_student.step()

                    # Uncomment to use the learning rate scheduler
                    # scheduler.step()
                    loss_list.append(loss.item())
                    # switch the student model validation to True
                    self.val_student = True
                    self.val_teacher = True

                if global_iter % pbar_interval == 0:
                    pbar.update(pbar_interval)
                    if len(loss_list) > 0:
                        print('epoch %d iter %5d, loss: %.3f\n' % (epoch, i_iter_train, np.mean(loss_list)))
                    else:
                        print('loss error')
                global_iter += 1

            # ----------------------------------------------------------------------#
            # Evaluation/validation
            with torch.no_grad():
                # Change teacher & student model to evaluation mode
                teacher_hist_list, student_hist_list = self.validate_uda(val_dataset_loader)

            # ----------------------------------------------------------------------#
            # Print validation mIoU and Loss
            print(f"--------------- epoch: {epoch} ----------------")
            # teacher validation
            if self.val_teacher:
                iou = per_class_iu(sum(teacher_hist_list))
                print('Teacher Validation per class iou: ')
                for class_name, class_iou in zip(self.unique_label_str, iou):
                    print('%s : %.2f%%' % (class_name, class_iou * 100))
                teacher_val_miou = np.nanmean(iou) * 100
                # save teacher model if performance is improved
                if self.teacher_best_val_miou < teacher_val_miou:
                    self.teacher_best_val_miou = teacher_val_miou
                    torch.save(self.teacher_model.state_dict(), self.teacher_model_save_path)
                print('Current teacher val miou is %.3f while the best  val miou is %.3f' % (
                    teacher_val_miou, self.teacher_best_val_miou))
                # del val_vox_label, val_grid, val_pt_fea

            # teacher validation
            if self.val_student:
                self.ema_update_now = True
                iou = per_class_iu(sum(student_hist_list))
                print('Student Validation per class iou: ')
                for class_name, class_iou in zip(self.unique_label_str, iou):
                    print('%s : %.2f%%' % (class_name, class_iou * 100))
                student_val_miou = np.nanmean(iou) * 100
                # save student model if performance is improved
                if self.student_best_val_miou < student_val_miou:
                    self.student_best_val_miou = student_val_miou
                print('Current Student val miou is %.3f while the best val miou is %.3f' % (
                    student_val_miou, self.student_best_val_miou))

                if self.ema_update_now:
                    # -------EMA ----------------#
                    # EMA: Student ---> Teacher
                    self._update_teacher_model()
                    print("--------------- EMA - Update Performed ----------------")
                    torch.save(self.teacher_model.state_dict(), self.teacher_model_save_path)
                    # switch the teacher model validation to True
                    self.val_teacher = True
                    # save student model
                    torch.save(self.student_model.state_dict(), self.student_model_save_path)

    #################################
