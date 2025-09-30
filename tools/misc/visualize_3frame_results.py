# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from os import path as osp

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet.apis import init_detector
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import cv2
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize PETRv2 3-Frame Adaptive results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--result', help='results file in pickle format')
    parser.add_argument(
        '--show-dir', help='directory where visualize results will be saved')
    parser.add_argument(
        '--frames',
        type=int,
        default=3,
        help='number of frames (default: 3)')
    parser.add_argument(
        '--sample-idx',
        type=int,
        default=None,
        help='sample index to visualize, default will visualize all samples')
    parser.add_argument(
        '--score-thr', 
        type=float, 
        default=0.4, 
        help='score threshold (default: 0.4)')
    parser.add_argument(
        '--adaptive-vis',
        action='store_true',
        help='visualize adaptive weights between frames')
    args = parser.parse_args()
    return args


def visualize_camera_with_boxes(img, 
                                boxes,
                                labels, 
                                scores, 
                                frame_idx=0, 
                                score_thr=0.4,
                                with_score=True):
    """Draw 3D boxes on image from camera view."""
    color_map = {
        0: (0, 255, 0),       # car: green
        1: (0, 0, 255),       # truck: blue
        2: (255, 0, 0),       # construction vehicle: red
        3: (255, 255, 0),     # bus: yellow
        4: (255, 0, 255),     # trailer: magenta
        5: (0, 255, 255),     # barrier: cyan
        6: (128, 0, 255),     # motorcycle: purple
        7: (128, 255, 0),     # bicycle: lime
        8: (255, 192, 203),   # pedestrian: pink
        9: (255, 165, 0)      # traffic cone: orange
    }
    
    img_copy = img.copy()
    
    # Draw frame indicator
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_copy, f'Frame {frame_idx}', (50, 50), font, 
                1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Filter by score threshold
    valid_inds = scores > score_thr
    boxes = boxes[valid_inds]
    labels = labels[valid_inds]
    scores = scores[valid_inds]
    
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        # Get box coordinates
        corners = box.corners.numpy()
        
        # Project 3D box to image plane
        corners_2d = []
        for corner in corners:
            corners_2d.append([corner[0], corner[1]])
        corners_2d = np.array(corners_2d, dtype=np.int32)
        
        # Get color based on class
        color = color_map.get(label, (255, 255, 255))
        
        # Draw box
        for j in range(4):
            cv2.line(img_copy, tuple(corners_2d[j]), tuple(corners_2d[(j+1)%4]), color, 2)
            cv2.line(img_copy, tuple(corners_2d[j+4]), tuple(corners_2d[(j+1)%4+4]), color, 2)
            cv2.line(img_copy, tuple(corners_2d[j]), tuple(corners_2d[j+4]), color, 2)
        
        # Add score text
        if with_score:
            label_text = f'cls: {label}, score: {score:.2f}'
            cv2.putText(img_copy, label_text, (corners_2d[0][0], corners_2d[0][1] - 5),
                        font, 0.5, color, 1, cv2.LINE_AA)
            
    return img_copy


def main():
    args = parse_args()

    if args.result is not None and \
            not args.result.endswith(('.pkl', '.pickle')):
        raise ValueError('The results file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True
    
    # Update configuration
    if args.frames:
        cfg.model.num_frames = args.frames
    
    # Set up show directory
    if args.show_dir is None:
        args.show_dir = 'visualization'
    mmcv.mkdir_or_exist(osp.abspath(args.show_dir))
    
    # Build dataset
    dataset = build_dataset(cfg.data.test)
    results = mmcv.load(args.result) if args.result else None
    
    # Build model
    if args.checkpoint:
        cfg.model.pretrained = None
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    # Get samples to visualize
    if args.sample_idx is not None:
        sample_idxs = [args.sample_idx]
    else:
        sample_idxs = list(range(len(dataset)))
        
    for idx in sample_idxs:
        try:
            print(f'Visualizing sample {idx}')
            
            # Get data and model predictions
            data = dataset[idx]
            file_name = osp.split(data['img_filename'][0])[-1].split('.')[0]
            
            # Create a subdirectory for each sample
            sample_dir = osp.join(args.show_dir, file_name)
            mmcv.mkdir_or_exist(sample_dir)
            
            # Visualize camera images with 3D boxes
            for cam_idx, img_path in enumerate(data['img_filename']):
                try:
                    img = mmcv.imread(img_path)
                    
                    # Get frame index based on camera index
                    # Assuming cameras are ordered like: frame0_cam0, frame0_cam1, ..., frame1_cam0, ...
                    frame_idx = cam_idx // (len(data['img_filename']) // args.frames)
                    
                    if results:
                        # Get prediction results for this sample
                        sample_results = results[idx]
                        
                        # Visualize detection results on this image
                        boxes = sample_results[0]
                        scores = sample_results[1]
                        labels = sample_results[2]
                        
                        # Draw boxes on image
                        vis_img = visualize_camera_with_boxes(
                            img, boxes, labels, scores, 
                            frame_idx=frame_idx, 
                            score_thr=args.score_thr
                        )
                    else:
                        # Just show the image without boxes
                        vis_img = img
                        
                    # Save the visualization
                    frame_name = f'frame{frame_idx}_cam{cam_idx % (len(data["img_filename"]) // args.frames)}.jpg'
                    mmcv.imwrite(vis_img, osp.join(sample_dir, frame_name))
                except Exception as e:
                    print(f"Error processing camera {cam_idx} for sample {idx}: {e}")
            
            # Visualize adaptive weights if available and requested
            if args.adaptive_vis and args.checkpoint:
                # TODO: Implement visualization of adaptive weights
                # This would require running inference and extracting the weights
                # from the temporal fusion module
                pass
                
            print(f'Visualization saved to {sample_dir}')
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue


if __name__ == '__main__':
    main() 