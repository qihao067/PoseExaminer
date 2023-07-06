import argparse
import os
import os.path as osp

import mmcv
import torch
import numpy as np

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


def parse_args():
    parser = argparse.ArgumentParser(description='mmhuman3d test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--work-dir', help='the dir to save evaluation results')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--name', default='none', help='LQH: adv testing') #LQH: adv testing
    parser.add_argument(
        '--domain_gap',
        action='store_true')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default='pa-mpjpe',
        help='evaluation metric, which depends on the dataset,'
        ' e.g., "pa-mpjpe" for H36M')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        default={},
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def prepare_anno(outputs):
    import numpy as np
    from pytorch3d import transforms
    # used to prepare annotation to bridge domain gap

    file_name = outputs[0]['image_path'][0].split('/')[2]
    if file_name == 'pw3d':
        gt_anno = np.load('./data/preprocessed_datasets/spin_pw3d_test.npz',allow_pickle=True)
    elif file_name == 'aist':
        gt_anno = np.load('./data/preprocessed_datasets/spin_aist_test_checked/spin_aist_test_ori.npz',allow_pickle=True)
    else:
        gt_anno = None

    __key_strict__ = np.array(False) 
    __keypoints_compressed__ = np.array(False) 
    __temporal_len__ = 0
    config = np.array(file_name)
    image_path = []
    global_orient = []
    betas = []
    body_pose = []

    for i in range(len(outputs)):
        temp = outputs[i]

        length = len(temp['image_path'])

        temp_smpl = transforms.quaternion_to_axis_angle(transforms.matrix_to_quaternion(torch.from_numpy(temp['smpl_pose']))).numpy()
        for k in range(length):
            assert gt_anno['image_path'][__temporal_len__] == temp['image_path'][k].split(file_name)[1][1:]
            
            image_path.append(temp['image_path'][k].split(file_name)[1][1:])
            global_orient.append(temp_smpl[k][0])
            body_pose.append(temp_smpl[k][1:])
            betas.append(temp['smpl_beta'][k])
            __temporal_len__ += 1
    image_path = np.array(image_path)
    global_orient = np.array(global_orient)
    body_pose = np.array(body_pose)
    betas = np.array(betas)
    bbox_xywh = gt_anno['bbox_xywh']
    meta = gt_anno['meta']

    smpl = np.array({'global_orient':global_orient,'body_pose':body_pose,'betas':betas})
    np.savez('./data/preprocessed_datasets/spin_'+file_name+'_domain.npz',__key_strict__=__key_strict__,__temporal_len__=__temporal_len__,__keypoints_compressed__=__keypoints_compressed__,image_path=image_path,bbox_xywh=bbox_xywh,config=config,smpl=smpl,meta=meta)

def main():
    args = parse_args()
    domain_gap = args.domain_gap
    with open('evaluation_results.txt','a') as f:
        f.write('##### Now testing checkpoints : [ '+str(str(args.checkpoint)) + ' ] \n')
        f.write('------------------- [ '+args.name + ' ] -------------------\n')

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    # the extra round_up data will be removed during gpu/cpu collect
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=False)

    # build the model and load checkpoint
    model = build_architecture(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if not distributed:
        if args.device == 'cpu':
            model = model.cpu()
        else:
            model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    if domain_gap:
        prepare_anno(outputs)

    rank, _ = get_dist_info()
    eval_cfg = cfg.get('evaluation', args.eval_options)
    eval_cfg.update(dict(metric=args.metrics))
    if rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        results = dataset.evaluate(outputs, args.work_dir, **eval_cfg)
        for k, v in results.items():
            print(f'\n{k} : {v:.2f}')
            with open('evaluation_results.txt','a') as f:
                f.write(f'{k} : {v:.2f} \n')
        with open('evaluation_results.txt','a') as f:
                f.write('---------------------------------------------------------\n'*2)

    if args.out and rank == 0:
        print(f'\nwriting results to {args.out}')
        mmcv.dump(results, args.out)


if __name__ == '__main__':
    main()
