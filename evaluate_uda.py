import argparse
import sys
import os

import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from tqdm import tqdm

from core import builder
from core.datasets.tuda_dataloader import get_label_name
from core.metric_util import fast_hist_crop, per_class_iu
from model_zoo import minkunet, spvcnn, spvnas_specialized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    # parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    parser.add_argument('--network', type=str, help='network type [student , teacher]')
    # parser.add_argument('--name', type=str, help='model name')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    x = args.network.lower()
    print(x, "teacher", x == "teacher")

    run_dir = configs.train_params.model_path + "/" + configs.model.name + "/" \
              + configs.train_params.source + "_" + configs.train_params.target

    feature = 'time' if configs.train_params.time else 'intensity'
    mode = 'uda' if configs.train_params.uda else 'inference'

    # run_dir = configs.train_params.model_path + "/" + configs.model.name + "/" \
    #           + configs.train_params.source + "_" + configs.train_params.target \
    #           + '_' + feature + '_' + mode

    run_dir = configs.train_params.model_path + "/" + configs.model.name + "/" \
              + configs.train_params.source + "_to_" + configs.train_params.target \
              + '_' + feature + '_' + mode \
              + f"_T{configs.train_params.past}_{configs.train_params.future}" \
              + f"_S{configs.test_params.past}_{configs.test_params.future}"

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

    modality_hypers = configs.test_params

    dataset = builder.make_dataset(model_configs=configs, modality_hypers=modality_hypers, ssl=configs.train_params.ssl)
    dataflow = {}
    for split in dataset:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset[split],
            num_replicas=dist.size(),
            rank=dist.rank(),
            shuffle=(split == 'train'))
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=configs.batch_size if split == 'train' else 1,
            sampler=sampler,
            num_workers=configs.workers_per_gpu,
            pin_memory=True,
            collate_fn=dataset[split].collate_fn)

    if args.network.lower() in "student":
        checkpoint = run_dir + '_best_student.pt'
    elif args.network.lower() in "teacher":
        checkpoint = run_dir + '_best_teacher.pt'
    else:
        checkpoint = run_dir + '/checkpoints/max-iou-test.pt'

    if 'spvnas' in configs.model.name:
        model = spvnas_specialized(configs.model.name, checkpoint, configs=configs)
    elif 'spvcnn' in configs.model.name:
        model = spvcnn(configs.model.name, checkpoint, configs=configs)
    elif 'mink' in configs.model.name:
        model = minkunet(configs.model.name, checkpoint, configs=configs)
    else:
        raise NotImplementedError

    if configs.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(),
            device_ids=[dist.local_rank()],
            find_unused_parameters=True)
    else:
        model = model.cuda()
    model.eval()

    label_name = get_label_name(configs.data.label_mapping)
    unique_label = np.asarray(sorted(list(label_name.keys())))[1:] - 1
    unique_label_str = [label_name[x] for x in unique_label + 1]
    print(unique_label)
    print(unique_label_str)

    hist_list = []
    for feed_dict in tqdm(dataflow['test'], desc='eval'):
        _inputs = {}
        for key, value in feed_dict.items():
            if ('name' not in key) and ('reference_idx' not in key):
                _inputs[key] = value.cuda()

        inputs = _inputs['lidar']

        outputs = model(inputs)

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
        outputs = torch.cat(_outputs).cpu().numpy()
        targets = torch.cat(_targets).cpu().numpy()
        hist_list.append(fast_hist_crop(outputs, targets, unique_label))
        file_name = feed_dict['file_name'][0].replace("lidar", "predictions").split("/")
        file_path = "/".join(file_name[:-1])
        name = file_name[-1]
        # print(os.path.join(file_path, name))
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        np.save(os.path.join(file_path, name), outputs)
        # print(f"outputs: {outputs.shape}, targets:, {targets.shape}")
    iou = per_class_iu(sum(hist_list))
    print(f'{args.network} Evaluation per class iou: ')
    for class_name, class_iou in zip(unique_label_str, iou):
        print(f"{class_name}, {class_iou * 100 :.2f}")
    val_miou = np.nanmean(iou) * 100
    print(f"{args.network} eval miou is {val_miou:.2f}")


if __name__ == '__main__':
    main()
