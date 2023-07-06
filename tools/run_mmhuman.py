import os.path as osp

import mmcv
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    get_dist_info,
    init_dist,
    load_checkpoint,
    wrap_fp16_model,
)

from mmhuman3d.apis import multi_gpu_test, single_gpu_test
from mmhuman3d.data.datasets import build_dataloader, build_dataset
from mmhuman3d.models.architectures.builder import build_architecture

import numpy as np


class PoseEva_mmhuman():
    def __init__(self, args, proc_id=-1, model_eva=False):
        super(PoseEva_mmhuman, self).__init__()
        
        self.args = args
        self.cfg = mmcv.Config.fromfile(self.args.config)

        self.cfg.data.workers_per_gpu = 1

        # set cudnn_benchmark
        if self.cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        self.cfg.data.test.test_mode = True

        # build the model and load checkpoint
        self.model = build_architecture(self.cfg.model)
        self.fp16_cfg = self.cfg.get('fp16', None)
        if self.fp16_cfg is not None:
            wrap_fp16_model(self.model)
        load_checkpoint(self.model, self.args.checkpoint, map_location='cpu')

        if self.args.device == 'cpu':
            self.model = self.model.cpu()
        else:
            self.model = MMDataParallel(self.model, device_ids=[0])
    
    def eva(self, proc_id=-1, iter=0, output_ver='min'):
        
        # build the dataloader
        # import pdb; pdb.set_trace()
        dataset = build_dataset(self.cfg.data.adv_test, proc_id=proc_id)

        # the extra round_up data will be removed during gpu/cpu collect
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=self.cfg.data.samples_per_gpu,
            workers_per_gpu=self.cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False,
            round_up=False)

        outputs = single_gpu_test(self.model, data_loader, proc_id, iter)

        rank, _ = get_dist_info()
        eval_cfg = self.cfg.get('evaluation', {})
        eval_cfg.update(dict(metric=self.args.metrics))
        if rank == 0:
            mmcv.mkdir_or_exist(osp.abspath(self.args.work_dir))
            results = dataset.evaluate(outputs, self.args.work_dir, **eval_cfg, output_ver=output_ver)
            # for k, v in results.items():
            #     print(f'\n{k} : {v:.2f}')
            if output_ver == 'min':
                return results['MPJPE-2D'].mean(axis=1).min(), results['MPJPE'].mean(axis=1).min(), results['PA-MPJPE'].mean(axis=1).min()
            elif output_ver == 'mean':
                return results['MPJPE-2D'].mean(), results['MPJPE'].mean(), results['PA-MPJPE'].mean()
            elif output_ver == 'all':
                per_c = (results['MPJPE'].mean(axis=1)<90).sum()/results['MPJPE'].shape[0]
                return results['MPJPE'].mean(), results['MPJPE'].mean(axis=1).min(), results['MPJPE'].mean(axis=1).max(), np.median(results['MPJPE'].mean(axis=1)), per_c
            else:
                return results['MPJPE-2D'], results['MPJPE'], results['MPJPE']


