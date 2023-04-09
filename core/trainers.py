import copy

import numpy as np
import torch
from torch import nn
from torch.cuda import amp
from torchpack.train import Trainer
from torchpack.utils.typing import Optimizer, Scheduler
from typing import Any, Callable, Dict

__all__ = ['SemanticKITTITrainer']


def yield_target_dataset_loader(n_epochs, target_train_dataset_loader):
    for e in range(n_epochs):
        for data_feed in target_train_dataset_loader:
            # print(data_feed)
            yield data_feed


class SemanticKITTITrainer(Trainer):

    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 optimizer: Optimizer,
                 scheduler: Scheduler,
                 num_workers: int,
                 warmup_epoch: int,
                 max_num_epochs: int,
                 target_dataset_loader: Dict[str, Any],
                 seed: int,
                 amp_enabled: bool = False,
                 ssl_mode: bool = False) -> None:
        self.model = model
        self.teacher_model = None
        self.start_uda = False
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_workers = num_workers
        self.warmup_epoch = warmup_epoch
        self.ema_update_now = False
        self.seed = seed
        self.amp_enabled = amp_enabled
        self.scaler = amp.GradScaler(enabled=self.amp_enabled)
        self.epoch_num = 1
        self.max_num_epochs = max_num_epochs
        self.ssl_mode = ssl_mode
        self.one_epoch_tracker = 0
        self.target_data_generator = yield_target_dataset_loader(self.max_num_epochs, target_dataset_loader)

    def calculate_loss(self, outputs, targets, _lcw=None):
        if self.start_uda:
            # print(outputs.size(), targets.size())
            raw_loss = self.criterion(outputs, targets)
            # print(f"raw_loss:{raw_loss.size()}, lcw: {lcw.F}")
            # extract the values/features from th SpraseTensor using ".F"
            w_loss = raw_loss * torch.squeeze(_lcw)
            loss = w_loss.mean()  # / 100.0
            # print(f"w_loss:{w_loss.size()}, loss: {loss}")
        elif self.ssl_mode:
            # print(outputs.size(), targets.size())
            raw_loss = self.criterion(outputs, targets)

            lcw = _lcw
            # print(f"raw_loss:{raw_loss.size()}, lcw: {lcw.F}")
            # extract the values/features from th SpraseTensor using ".F"
            lcw = lcw.F
            w_loss = raw_loss * torch.squeeze(lcw)
            loss = w_loss.mean() / 100.0
            # print(f"w_loss:{w_loss.size()}, loss: {loss}")
        else:
            # print(outputs.size(), targets.size())
            loss = self.criterion(outputs, targets)

            # print(f"loss.size:{loss.size()}, loss: {loss.item()}")
        return loss

    # updating teacher model weights
    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.996):
        student_model_dict = self.model.state_dict()

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

    def _before_epoch(self) -> None:
        self.model.train()
        self.dataflow.sampler.set_epoch(self.epoch_num - 1)

        self.dataflow.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id)

    # TODO: add the UDA training part bellow
    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        _inputs = {}
        for key, value in feed_dict.items():
            if ('name' not in key) and ('reference_idx' not in key):
                _inputs[key] = value.cuda()
        # print(f"_inputs: {_inputs}")

        # print(f"self.epoch_num: {self.epoch_num}, self.local_step {self.local_step}")
        if (self.epoch_num == self.warmup_epoch) and (self.local_step == 1):
            # create teacher model
            # self.model = list(self.model)  # where attribute was dict_keys
            self.teacher_model = copy.deepcopy(self.model)
            student_model_dict = self.model.state_dict()
            teacher_model_dict = self.teacher_model.state_dict()
            print(f"1student_model_dict: {student_model_dict['stem.0.kernel']}")
            print(f"1teacher_model_dict: {teacher_model_dict['stem.0.kernel']}")
            # change teacher model to eval mode
            self.teacher_model.eval()
            # start uda mode (since the base model is trained for warmup epoch)
            self.start_uda = True
            print("\n\n-------------------------UDA Started----------------------------")

        elif (self.epoch_num > self.warmup_epoch) and (self.local_step == 1):
            self.ema_update_now = True

        if self.start_uda:
            inputs = _inputs['lidar']
            targets = feed_dict['targets'].F.long().cuda(non_blocking=True)

            with amp.autocast(enabled=self.amp_enabled):
                # Student forward pass on source data
                outputs = self.model(inputs)

                if outputs.requires_grad:

                    if self.ema_update_now:
                        # -------EMA ----------------#
                        self.teacher_model.eval()
                        # EMA: Student ---> Teacher
                        student_model_dict = self.model.state_dict()
                        teacher_model_dict = self.teacher_model.state_dict()
                        print(f"student_model_dict: {student_model_dict['stem.0.kernel']}")
                        print(f"teacher_model_dict: {teacher_model_dict['stem.0.kernel']}")
                        self._update_teacher_model()
                        print("--------------- EMA - Update Performed ----------------")
                        self.ema_update_now = False

                    feed_dict_target_data = next(self.target_data_generator)
                    _inputs_targets = {}
                    for key, value in feed_dict_target_data.items():
                        if ('name' not in key) and ('reference_idx' not in key):
                            _inputs_targets[key] = value.cuda()
                    inputs_target_data = _inputs_targets['lidar']
                    # targets_target_data = feed_dict_target_data['targets'].F.long().cuda(non_blocking=True)
                    # inference/pseudo labeling of target data
                    target_prediction = self.teacher_model(inputs_target_data)
                    # pseudo label generated by teacher model
                    pseudo_label = torch.squeeze(torch.argmax(target_prediction, dim=1))
                    # calculate weight/confidence of each pseudo label
                    predict_probability = torch.nn.functional.softmax(target_prediction, dim=1)
                    # pseudo label confidence ---> lcw
                    pseudo_labels_prob_lcw, predict_prob_ind = predict_probability.max(dim=1)
                    # multiply by 100
                    pseudo_labels_prob_lcw = pseudo_labels_prob_lcw
                    # Create a lcw tensor of ones for the source data and multiply by 100
                    source_lcw = torch.ones_like(targets)
                    ###

                    # Student forward pass on target data
                    outputs_target_data = self.model(inputs_target_data)
                    # print(f"inpust: {inputs.F}")

                    loss_source_data = self.calculate_loss(outputs, targets, source_lcw)
                    loss_target_data = self.calculate_loss(outputs_target_data, pseudo_label, pseudo_labels_prob_lcw)
                    # print(loss_source_data, loss_target_data)

                    loss = loss_source_data + loss_target_data

            # if outputs.requires_grad:
            #     self.summary.add_scalar('loss', loss.item())
            #
            #     self.optimizer.zero_grad()
            #     self.scaler.scale(loss).backward()
            #     self.scaler.step(self.optimizer)
            #     self.scaler.update()
            #     self.scheduler.step()
            # else:
            #     invs = feed_dict['inverse_map']
            #     all_labels = feed_dict['targets_mapped']
            #     _outputs = []
            #     _targets = []
            #     for idx in range(invs.C[:, -1].max() + 1):
            #         cur_scene_pts = (inputs.C[:, -1] == idx).cpu().numpy()
            #         cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
            #         cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
            #         outputs_mapped = outputs[cur_scene_pts][cur_inv].argmax(1)
            #         targets_mapped = all_labels.F[cur_label]
            #         _outputs.append(outputs_mapped)
            #         _targets.append(targets_mapped)
            #     outputs = torch.cat(_outputs, 0)
            #     targets = torch.cat(_targets, 0)

        else:
            inputs = _inputs['lidar']
            targets = feed_dict['targets'].F.long().cuda(non_blocking=True)

            with amp.autocast(enabled=self.amp_enabled):
                outputs = self.model(inputs)
                # print(f"inpust: {inputs.F}")

                if outputs.requires_grad:
                    loss = self.calculate_loss(outputs, targets)
                    # print(loss)

        if outputs.requires_grad:
            self.summary.add_scalar('loss', loss.item())

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
        else:
            invs = feed_dict['inverse_map']
            all_labels = feed_dict['targets_mapped']
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
            outputs = torch.cat(_outputs, 0)
            targets = torch.cat(_targets, 0)
            # equal = (outputs_mapped.cpu().numpy() == targets_mapped.cpu().numpy()).astype(np.float)
            # print(f"equal->: {equal}")
            # print(f"mean: {np.mean(equal)}")

        return {'outputs': outputs, 'targets': targets}

    def _after_epoch(self) -> None:
        self.model.eval()

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = {}
        state_dict['model'] = self.model.state_dict()
        state_dict['scaler'] = self.scaler.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict['model'])
        self.scaler.load_state_dict(state_dict.pop('scaler'))
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])

    def _load_previous_checkpoint(self, checkpoint_path: str) -> None:
        pass
