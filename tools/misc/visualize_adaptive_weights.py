# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from os import path as osp
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet.datasets.pipelines import Compose


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize adaptive weights in PETRv2 3-Frame model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--show-dir', help='directory where visualize results will be saved')
    parser.add_argument(
        '--frames',
        type=int,
        default=3,
        help='number of frames (default: 3)')
    parser.add_argument(
        '--sample-idx',
        type=str,
        default=None,
        help='sample index to visualize, default will visualize all samples')
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='number of samples to visualize (default: 10)')
    args = parser.parse_args()
    return args


class AdaptiveWeightHook:
    """Hook to capture adaptive weights during model forward pass."""
    
    def __init__(self):
        self.weights = []
        self.hooks = []
        
    def register_hooks(self, model):
        # Find all temporal fusion modules
        for name, module in model.named_modules():
            if 'temporal_fusion' in name and hasattr(module, 'adaptive_weights'):
                # Define hook function
                def hook_fn(module, input, output, module_name=name):
                    if hasattr(module, 'adaptive_weights'):
                        # Extract and save the adaptive weights
                        self.weights.append({
                            'module': module_name,
                            'weights': module.adaptive_weights.detach().cpu().numpy()
                        })
                
                # Register hook
                handle = module.register_forward_hook(hook_fn)
                self.hooks.append(handle)
                
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def plot_adaptive_weights(weights_data, show_dir, sample_idx):
    """Plot the adaptive weights distribution."""
    
    output_dir = osp.join(show_dir, f'sample_{sample_idx}')
    mmcv.mkdir_or_exist(output_dir)
    
    for idx, data in enumerate(weights_data):
        module_name = data['module']
        weights = data['weights']
        
        # Prepare data for visualization
        if len(weights.shape) == 3:  # [B, Q, T]
            batch_size, num_queries, num_frames = weights.shape
            
            # Plot heatmap for each batch
            for b in range(batch_size):
                # Create heatmap of weights
                plt.figure(figsize=(10, 8))
                ax = sns.heatmap(weights[b], cmap="YlGnBu", 
                                vmin=0, vmax=1.0,
                                xticklabels=[f'Frame {i}' for i in range(num_frames)],
                                yticklabels=np.arange(num_queries))
                plt.title(f'Adaptive Weights Distribution - {module_name} (Batch {b})')
                plt.xlabel('Frames')
                plt.ylabel('Query Index')
                plt.tight_layout()
                
                # Save figure
                plt.savefig(osp.join(output_dir, f'{module_name.replace(".", "_")}_batch{b}.png'))
                plt.close()
                
            # Plot average weight per frame
            avg_weights = np.mean(weights, axis=(0, 1))
            plt.figure(figsize=(8, 6))
            plt.bar(np.arange(num_frames), avg_weights)
            plt.xlabel('Frame Index')
            plt.ylabel('Average Weight')
            plt.title(f'Average Adaptive Weight Per Frame - {module_name}')
            plt.xticks(np.arange(num_frames), [f'Frame {i}' for i in range(num_frames)])
            plt.ylim(0, 1.0)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(osp.join(output_dir, f'{module_name.replace(".", "_")}_avg_weights.png'))
            plt.close()
            
            # Create summary statistics
            stats = {
                'frame': [],
                'min': [],
                'max': [],
                'mean': [],
                'std': []
            }
            
            for f in range(num_frames):
                frame_weights = weights[:, :, f].flatten()
                stats['frame'].append(f)
                stats['min'].append(np.min(frame_weights))
                stats['max'].append(np.max(frame_weights))
                stats['mean'].append(np.mean(frame_weights))
                stats['std'].append(np.std(frame_weights))
            
            # Save statistics
            stats_df = pd.DataFrame(stats)
            stats_df.to_csv(osp.join(output_dir, f'{module_name.replace(".", "_")}_stats.csv'), index=False)
        
        elif len(weights.shape) == 2:  # [B, T]
            batch_size, num_frames = weights.shape
            
            # Plot bar chart for each batch
            for b in range(batch_size):
                plt.figure(figsize=(8, 6))
                plt.bar(np.arange(num_frames), weights[b])
                plt.xlabel('Frame Index')
                plt.ylabel('Weight Value')
                plt.title(f'Adaptive Weights - {module_name} (Batch {b})')
                plt.xticks(np.arange(num_frames), [f'Frame {i}' for i in range(num_frames)])
                plt.ylim(0, 1.0)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                # Save figure
                plt.savefig(osp.join(output_dir, f'{module_name.replace(".", "_")}_batch{b}.png'))
                plt.close()
                
            # Plot average weights
            avg_weights = np.mean(weights, axis=0)
            plt.figure(figsize=(8, 6))
            plt.bar(np.arange(num_frames), avg_weights)
            plt.xlabel('Frame Index')
            plt.ylabel('Average Weight')
            plt.title(f'Average Adaptive Weight Per Frame - {module_name}')
            plt.xticks(np.arange(num_frames), [f'Frame {i}' for i in range(num_frames)])
            plt.ylim(0, 1.0)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(osp.join(output_dir, f'{module_name.replace(".", "_")}_avg_weights.png'))
            plt.close()
            
            # Create summary statistics
            stats = {
                'frame': list(range(num_frames)),
                'mean': avg_weights.tolist(),
                'std': np.std(weights, axis=0).tolist()
            }
            
            # Save statistics
            stats_df = pd.DataFrame(stats)
            stats_df.to_csv(osp.join(output_dir, f'{module_name.replace(".", "_")}_stats.csv'), index=False)


def main():
    args = parse_args()

    # Set up show directory
    if args.show_dir is None:
        args.show_dir = 'adaptive_weight_visualization'
    mmcv.mkdir_or_exist(osp.abspath(args.show_dir))
    
    # Load config and model
    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True
    
    # Update configuration for number of frames
    if args.frames:
        cfg.model.num_frames = args.frames
    
    # Build dataset
    dataset = build_dataset(cfg.data.test)
    
    # Build model
    cfg.model.pretrained = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cuda()
    model.eval()
    
    # Create test pipeline
    test_pipeline = cfg.data.test.pipeline
    test_pipeline = Compose(test_pipeline)
    
    # Get samples to visualize
    if args.sample_idx is not None:
        sample_idxs = [int(args.sample_idx)]
    else:
        # Choose a subset of samples
        sample_idxs = list(range(min(args.num_samples, len(dataset))))
    
    # Register hooks to capture adaptive weights
    weight_hook = AdaptiveWeightHook()
    weight_hook.register_hooks(model)
    
    # Process each sample
    for idx in sample_idxs:
        print(f'Visualizing adaptive weights for sample {idx}')
        
        # Get data
        data = dataset[idx]
        data = test_pipeline(data)
        
        # Convert data to device
        data = collate([data], samples_per_gpu=1)
        data = scatter(data, ['cuda:0'])[0]
        
        # Clear previous weights
        weight_hook.weights = []
        
        # Forward pass to capture weights
        with torch.no_grad():
            model.forward_dummy(**data)
        
        # Visualize weights
        if weight_hook.weights:
            plot_adaptive_weights(weight_hook.weights, args.show_dir, idx)
            print(f'Visualization saved to {osp.join(args.show_dir, f"sample_{idx}")}')
        else:
            print('No adaptive weights were captured. Check that your model has temporal fusion modules with adaptive weights.')
    
    # Remove hooks
    weight_hook.remove_hooks()
    print('Visualization complete.')


if __name__ == '__main__':
    main() 