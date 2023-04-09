import json
import os
import sys
from urllib.request import urlretrieve

import torch
from torchpack import distributed as dist

from core.models.basemodels.minkunet import MinkUNet
from core.models.basemodels.spvcnn import SPVCNN
from core.models.basemodels.spvnas import SPVNAS

__all__ = ['spvnas_specialized', 'minkunet', 'spvcnn']


def download_url(url, model_dir='~/.torch/', overwrite=False):
    target_dir = url.split('/')[-1]
    model_dir = os.path.expanduser(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_dir = os.path.join(model_dir, target_dir)
    cached_file = model_dir
    if not os.path.exists(cached_file) or overwrite:
        sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
        urlretrieve(url, cached_file)
    return cached_file


def spvnas_specialized(net_id, checkpoint, pretrained=True, configs=None, **kwargs):

    model = SPVCNN(num_classes=configs.data.num_classes,
                   macro_depth_constraint=1,
                   pres=configs.dataset.voxel_size,
                   vres=configs.dataset.voxel_size).to(
            'cuda:%d'
            % dist.local_rank() if torch.cuda.is_available() else 'cpu')

    if pretrained:

        init = torch.load(checkpoint,
                          map_location='cuda:%d' % dist.local_rank()
                          if torch.cuda.is_available() else 'cpu')['model']
        model.load_state_dict(init)
    return model


def spvnas_supernet(net_id, pretrained=True, **kwargs):
    url_base = 'https://hanlab.mit.edu/files/SPVNAS/spvnas_supernet/'
    net_config = json.load(
        open(
            download_url(url_base + net_id + '/net.config',
                         model_dir='.torch/spvnas_supernet/%s/' % net_id)))

    model = SPVNAS(
        net_config['num_classes'],
        macro_depth_constraint=net_config['macro_depth_constraint'],
        pres=net_config['pres'],
        vres=net_config['vres']).to(
            'cuda:%d'
            % dist.local_rank() if torch.cuda.is_available() else 'cpu')

    if pretrained:
        init = torch.load(download_url(url_base + net_id + '/init',
                                       model_dir='.torch/spvnas_supernet/%s/'
                                       % net_id),
                          map_location='cuda:%d' % dist.local_rank()
                          if torch.cuda.is_available() else 'cpu')['model']
        model.load_state_dict(init)
    return model


def minkunet(net_id, checkpoint, pretrained=True, configs=None, **kwargs):
    if 'cr' in configs.model:
        cr = configs.model.cr
    else:
        cr = 1.0
    model = MinkUNet(num_classes=configs.data.num_classes,
                     cr=cr).to(
            'cuda:%d'
            % dist.local_rank() if torch.cuda.is_available() else 'cpu')

    print(checkpoint)
    if pretrained:

        init = torch.load(checkpoint,
                          map_location='cuda:%d' % dist.local_rank()
                          if torch.cuda.is_available() else 'cpu')#['model']
        model.load_state_dict(init)

    return model


def spvcnn(net_id, checkpoint, pretrained=True, configs=None, **kwargs):
    model = SPVCNN(num_classes=configs.data.num_classes,
                   cr=configs.model.cr,
                   pres=configs.dataset.voxel_size,
                   vres=configs.dataset.voxel_size).to(
            'cuda:%d'
            % dist.local_rank() if torch.cuda.is_available() else 'cpu')

    if pretrained:
        init = torch.load(checkpoint,
                          map_location='cuda:%d' % dist.local_rank()
                          if torch.cuda.is_available() else 'cpu')['model']
        model.load_state_dict(init)

    return model
