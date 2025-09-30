# 3-Frame Adaptive Model Visualization Tools

This directory contains visualization tools specifically designed for the PETRv2 3-Frame Adaptive model. These tools help visualize detector outputs and adaptive weighting mechanisms.

## Available Tools

### 1. `visualize_3frame_results.py`

This script visualizes 3D detection results from the 3-Frame Adaptive model, showing bounding boxes on camera images with frame-specific information.

#### Usage:

```bash
python tools/misc/visualize_3frame_results.py \
    CONFIG_FILE \
    CHECKPOINT_FILE \
    RESULT_FILE \
    [--show-dir SHOW_DIR] \
    [--frames FRAMES] \
    [--sample-idx SAMPLE_IDX] \
    [--score-thr SCORE_THR] \
    [--adaptive-vis]
```

#### Arguments:

- `CONFIG_FILE`: Path to the model configuration file
- `CHECKPOINT_FILE`: Path to the model checkpoint file
- `RESULT_FILE`: Path to the pickle file containing detection results
- `--show-dir`: Directory to save visualization results (default: ./vis_results)
- `--frames`: Number of frames to visualize (default: 3)
- `--sample-idx`: Specific sample index to visualize (default: visualize all samples)
- `--score-thr`: Score threshold for filtering detections (default: 0.3)
- `--adaptive-vis`: Flag to enable visualizing adaptive weights if available

#### Example:

```bash
python tools/misc/visualize_3frame_results.py \
    configs/petrv2_3frame_adaptive/petrv2_vovnet_3frame_adaptive_nusc.py \
    work_dirs/petrv2_3frame_adaptive/latest.pth \
    work_dirs/petrv2_3frame_adaptive/results.pkl \
    --show-dir vis_results_3frame \
    --frames 3 \
    --score-thr 0.3
```

### 2. `visualize_adaptive_weights.py`

This script specifically visualizes the adaptive weights used in the temporal fusion modules of the 3-Frame Adaptive model. It creates heatmaps, bar charts, and statistical summaries of the weights assigned to each frame.

#### Usage:

```bash
python tools/misc/visualize_adaptive_weights.py \
    CONFIG_FILE \
    CHECKPOINT_FILE \
    [--show-dir SHOW_DIR] \
    [--frames FRAMES] \
    [--sample-idx SAMPLE_IDX] \
    [--num-samples NUM_SAMPLES]
```

#### Arguments:

- `CONFIG_FILE`: Path to the model configuration file
- `CHECKPOINT_FILE`: Path to the model checkpoint file
- `--show-dir`: Directory to save visualization results (default: ./adaptive_weight_visualization)
- `--frames`: Number of frames (default: 3)
- `--sample-idx`: Specific sample index to visualize (default: visualize a subset of samples)
- `--num-samples`: Number of samples to visualize (default: 10)

#### Example:

```bash
python tools/misc/visualize_adaptive_weights.py \
    configs/petrv2_3frame_adaptive/petrv2_vovnet_3frame_adaptive_nusc.py \
    work_dirs/petrv2_3frame_adaptive/latest.pth \
    --show-dir adaptive_weights_vis \
    --num-samples 5
```

## Output

### 3D Bounding Box Visualization (`visualize_3frame_results.py`)

The script creates visualizations with 3D bounding boxes projected onto camera images, with different colors indicating different frames. The output will be saved in the specified `--show-dir` directory, organized by sample and camera view.

### Adaptive Weight Visualization (`visualize_adaptive_weights.py`)

The script generates several visualization types:

1. **Heatmaps**: Shows the distribution of adaptive weights across queries and frames
2. **Bar Charts**: Shows the average weights assigned to each frame
3. **CSV Statistics**: Summary statistics (min, max, mean, std) of weights for each frame

The visualizations are saved in the specified `--show-dir` directory, organized by sample.

## Requirements

Both scripts require the following dependencies:
- mmcv
- torch
- numpy
- matplotlib
- seaborn (for heatmap visualization)
- pandas (for statistics)

Make sure the environment has these dependencies installed before running the scripts. 