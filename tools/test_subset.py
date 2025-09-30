#!/usr/bin/env python

import argparse
import os
import mmcv
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
import numpy as np
from mmdet3d.datasets import NuScenesDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test subset of 3D detection model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--subset-scenes',
        help='File listing scene tokens to evaluate, one per line')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        default='bbox',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument(
        '--show-dir', help='directory where visualizations will be saved')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


# Modify NuScenesDataset to fix the sample mismatch issue
def patch_nuscenes_dataset():
    """Monkey patch the NuScenesDataset to handle sample mismatch during evaluation."""
    original_evaluate = NuScenesDataset.evaluate
    
    def patched_evaluate(self, results, logger=None, **kwargs):
        """Patched evaluate method to fix sample mismatch."""
        print(f"Running patched NuScenes evaluation")
        
        # Make sure prediction and ground truth samples match
        if hasattr(self, 'data_infos') and results:
            data_sample_tokens = set([info['token'] for info in self.data_infos])
            result_sample_tokens = set(results[0].keys() if isinstance(results[0], dict) else [])
            
            # Print debugging info
            print(f"Dataset samples: {len(data_sample_tokens)}")
            print(f"Result samples: {len(result_sample_tokens)}")
            
            # Find mismatches
            missing_in_results = data_sample_tokens - result_sample_tokens
            extra_in_results = result_sample_tokens - data_sample_tokens
            
            if missing_in_results:
                print(f"Warning: {len(missing_in_results)} samples in dataset but missing in results")
                
                # Create empty predictions for missing samples
                for token in missing_in_results:
                    if isinstance(results[0], dict):
                        for result_dict in results:
                            result_dict[token] = []  # Empty prediction
            
            if extra_in_results:
                print(f"Warning: {len(extra_in_results)} extra samples in results not in dataset")
                # Remove extra samples from results
                if isinstance(results[0], dict):
                    for result_dict in results:
                        for token in extra_in_results:
                            if token in result_dict:
                                del result_dict[token]
        
        # Call the original evaluate function
        return original_evaluate(self, results, logger, **kwargs)
    
    # Apply the patch
    NuScenesDataset.evaluate = patched_evaluate
    print("NuScenesDataset.evaluate patched for sample consistency")

# Apply patches when this module is loaded
patch_nuscenes_dataset()

def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    # Add custom dataset patch for scene token filtering
    if args.subset_scenes:
        with open(args.subset_scenes, 'r') as f:
            scene_tokens = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Using subset of {len(scene_tokens)} scenes for evaluation")
        
        # Store scene tokens to filter samples during dataset creation
        if isinstance(cfg.data.test, dict):
            cfg.data.test.scene_tokens = scene_tokens
        elif isinstance(cfg.data.test, list):
            for ds_cfg in cfg.data.test:
                ds_cfg.scene_tokens = scene_tokens
    
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    # Allow valid_ratio < 1.0 for subset testing
    if isinstance(cfg.data.test, dict):
        cfg.data.test.valid_ratio = 1.0 if not hasattr(cfg.data.test, 'valid_ratio') else cfg.data.test.valid_ratio
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.valid_ratio = 1.0 if not hasattr(ds_cfg, 'valid_ratio') else ds_cfg.valid_ratio
    
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # After building dataset, filter data_infos by scene_token
    if hasattr(dataset, 'data_infos') and args.subset_scenes:
        orig_len = len(dataset.data_infos)
        scene_tokens = set(scene_tokens)
        
        # Filter data_infos to only include samples from specified scenes
        dataset.data_infos = [
            info for info in dataset.data_infos 
            if info.get('scene_token') in scene_tokens
        ]
        print(f"Filtered dataset from {orig_len} to {len(dataset.data_infos)} samples based on scene tokens")

    # build the model and load checkpoint
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
        
    # Added print to debug model and dataset
    print(f"Model classes: {model.CLASSES}")
    print(f"Dataset samples: {len(dataset)}")
        
    # Add a custom pre-evaluation step to ensure consistent samples
    if hasattr(dataset, 'data_infos') and hasattr(model, 'test_list'):
        # Make model.test_list match dataset.data_infos
        model.test_list = [info['token'] for info in dataset.data_infos]
        print(f"Aligned test samples: {len(model.test_list)}")

    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' not in checkpoint.get('meta', {}):
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main() 