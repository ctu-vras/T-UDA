from typing import Any, Dict

import numpy as np
import torch
from torchpack import distributed as dist
from torchpack.callbacks.callback import Callback

__all__ = ['MeanIoU']


class MeanIoU(Callback):

    def __init__(self,
                 num_classes: int,
                 ignore_label: int,
                 *,
                 output_tensor: str = 'outputs',
                 target_tensor: str = 'targets',
                 name: str = 'iou') -> None:
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.name = name
        self.output_tensor = output_tensor
        self.target_tensor = target_tensor

    def _before_epoch(self) -> None:
        self.total_seen = np.zeros(self.num_classes)
        self.total_correct = np.zeros(self.num_classes)
        self.total_positive = np.zeros(self.num_classes)

    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        outputs = output_dict[self.output_tensor]
        targets = output_dict[self.target_tensor]
        # print(f"targets:{targets.size()}, outputs:{outputs.size()}")
        targets = torch.squeeze(targets)
        # print(f"targets:{targets.size()}, outputs:{outputs.size()}")
        # print(f"targets:{targets}, outputs:{outputs}")
        outputs = outputs[targets != self.ignore_label]
        targets = targets[targets != self.ignore_label]
        # print("output and target", outputs, targets)

        for i in range(self.num_classes):
            self.total_seen[i] += torch.sum(targets == i).item()
            self.total_correct[i] += torch.sum((targets == i)
                                               & (outputs == targets)).item()
            self.total_positive[i] += torch.sum(outputs == i).item()

        # print(self.total_seen, self.total_correct, self.total_positive)

    def _after_epoch(self) -> None:
        for i in range(self.num_classes):
            self.total_seen[i] = dist.allreduce(self.total_seen[i],
                                                reduction='sum')
            self.total_correct[i] = dist.allreduce(self.total_correct[i],
                                                   reduction='sum')
            self.total_positive[i] = dist.allreduce(self.total_positive[i],
                                                    reduction='sum')

        ious = []

        for i in range(self.num_classes):
            if self.total_seen[i] == 0:
                ious.append(1)
            else:
                cur_iou = self.total_correct[i] / (self.total_seen[i]
                                                   + self.total_positive[i]
                                                   - self.total_correct[i])
                ious.append(np.round(cur_iou, 2))

        miou = np.mean(ious)
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'summary'):
            self.trainer.summary.add_scalar(self.name, miou * 100)
        else:
            print(f"IoUs: {ious}")
            print(f"mIoUs: {np.round(miou, 2)}")