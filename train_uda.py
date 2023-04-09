# -*- coding:utf-8 -*-
# author: Awet

import argparse
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
import yaml
from torchpack import distributed as dist
from torchpack.callbacks import InferenceRunner, MaxSaver, Saver
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger

from core import builder
from core.callbacks import MeanIoU
from core.datasets.tuda_dataloader import get_label_name
from core.trainer_function import TUDATrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    feature = 'time' if configs.train_params.time else 'intensity'
    mode = 'uda' if configs.train_params.uda else 'inference'

    run_dir = configs.train_params.model_path + "/" + configs.model.name + "/" \
              + configs.train_params.source + "_to_" + configs.train_params.target \
              + '_' + feature + '_' + mode \
              + f"_T{configs.train_params.past}_{configs.train_params.future}" \
              + f"_S{configs.test_params.past}_{configs.test_params.future}"
    print(run_dir)

    # NB: ignored class
    label_name = get_label_name(configs.data.label_mapping)
    unique_label = np.asarray(sorted(list(label_name.keys())))[1:] - 1
    unique_label_str = [label_name[x] for x in unique_label + 1]

    if configs.distributed:
        dist.init()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    if run_dir is None:
        run_dir = auto_set_run_dir()
    else:
        set_run_dir(run_dir)

    logger.info(' '.join([sys.executable] + sys.argv))
    logger.info(f'Experiment started: "{run_dir}".' + '\n' + f'{configs}')

    # seed
    if ('seed' not in configs.train) or (configs.train.seed is None):
        configs.train.seed = torch.initial_seed() % (2 ** 32 - 1)

    seed = configs.train.seed + dist.rank(
    ) * configs.workers_per_gpu * configs.num_epochs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    modality_hypers = configs.train_params
    dataset = builder.make_dataset(model_configs=configs, modality_hypers=modality_hypers, ssl=configs.ssl)

    dataflow = {}
    for split in dataset:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset[split],
            num_replicas=dist.size(),
            rank=dist.rank(),
            shuffle=(split == 'train'))
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=configs.batch_size,
            sampler=sampler,
            num_workers=configs.workers_per_gpu,
            pin_memory=True,
            collate_fn=dataset[split].collate_fn)

    model_student = builder.make_model().cuda()
    model_teacher = builder.make_model().cuda()
    if configs.distributed:
        model_student = torch.nn.parallel.DistributedDataParallel(
            model_student, device_ids=[dist.local_rank()], find_unused_parameters=True)
        model_teacher = torch.nn.parallel.DistributedDataParallel(
            model_teacher, device_ids=[dist.local_rank()], find_unused_parameters=True)

    criterion = builder.make_criterion(ssl_mode=configs.ssl)
    optimizer_student = builder.make_optimizer(model_student)
    optimizer_teacher = builder.make_optimizer(model_teacher)
    scheduler_student = builder.make_scheduler(optimizer_student)
    scheduler_teacher = builder.make_scheduler(optimizer_teacher)

    trainer = TUDATrainer(model_student=model_student,
                          model_teacher=model_teacher,
                          criterion=criterion,
                          optimizer_teacher=optimizer_teacher,
                          optimizer_student=optimizer_student,
                          scheduler_student=scheduler_student,
                          scheduler_teacher=scheduler_teacher,
                          num_workers=configs.workers_per_gpu,
                          warmup_epoch=configs.train_params.ema_start_epoch,
                          max_num_epochs=configs.num_epochs,
                          target_dataset_loader=dataflow['pseudo'],
                          num_classes=configs.data.num_classes,
                          ignore_label=configs.data.ignore_label,
                          unique_label=unique_label,
                          unique_label_str=unique_label_str,
                          run_dir=run_dir,
                          seed=seed,
                          amp_enabled=configs.amp_enabled,
                          ssl_mode=configs.ssl)
    print(f"configs.data.ignore_label: {configs.data.ignore_label}")
    trainer.uda_fit(dataflow['train'],
                    dataflow['pseudo'],
                    dataflow['val'],
                    test_loader=None)


if __name__ == '__main__':
    main()
