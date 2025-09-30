#!/bin/bash
#SBATCH --account=ywang234_1595
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --job-name=petr_viz
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# Create log directory
mkdir -p slurm_logs

# Configuration
WORK_DIR="/scratch1/ayushgoy/work_dir/4frame_adaptive"
PETR_DIR="/project2/ywang234_1595/petr_v2/ayushgoy/PETR"
CONFIG_FILE="${PETR_DIR}/projects/configs/petrv2/petrv2_3frame_adaptive.py"
RESULTS_DIR="${WORK_DIR}/results_eval/pts_bbox"
VIS_DIR="${WORK_DIR}/visualizations/detection_results"

# Create necessary directories
mkdir -p ${RESULTS_DIR}
mkdir -p ${VIS_DIR}

# Environment setup
source /home1/ayushgoy/.bashrc
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true

module purge
module load conda
module load legacy/CentOS7  
module load gcc/9.2.0
module load cuda/11.2.2

eval "$(conda shell.bash hook)"
conda activate 677_project

# Set library path
export LD_LIBRARY_PATH=~/.conda/envs/677_project/lib:$LD_LIBRARY_PATH 

# Change to PETR directory
cd ${PETR_DIR}

# --- INSTALL/VERIFY MMDETECTION & MMDETECTION3D --- 
echo "Checking/Installing mmdetection and mmdetection3d..."

# Check if mmdetection is installed and importable
if python -c "import mmdet" &> /dev/null; then
    echo "mmdetection found."
else
    echo "mmdetection not found or importable. Installing..."
    if [ ! -d "mmdetection" ]; then
        git clone https://github.com/open-mmlab/mmdetection.git mmdetection
    fi
    cd mmdetection
    git checkout v2.24.1 # Version compatible with mmdet3d v0.17.1
    pip install -e .
    cd ..
fi

# Check if mmdetection3d is installed and importable
if python -c "import mmdet3d" &> /dev/null; then
    echo "mmdetection3d found."
else
    echo "mmdetection3d not found or importable. Installing..."
    if [ ! -d "mmdetection3d" ]; then
        git clone https://github.com/open-mmlab/mmdetection3d.git mmdetection3d
    fi
    cd mmdetection3d
    git checkout v0.17.1 # Version used by PETR
    pip install -e .
    cd ..
fi

# CRITICAL: Set PYTHONPATH to include the local installs
export PYTHONPATH=${PETR_DIR}/mmdetection:${PETR_DIR}/mmdetection3d:${PYTHONPATH}

echo "PYTHONPATH set to: $PYTHONPATH"

# Final check
python -c "
import sys
print('Python path:')
for p in sys.path:
    print('  -', p)
try:
    import mmdet
    print('mmdet version:', mmdet.__version__)
    import mmdet3d
    print('mmdet3d version:', mmdet3d.__version__)
    print('Import successful!')
except ImportError as e:
    print('ERROR: Import failed:', e)
    exit(1)
"

# Find the latest checkpoint
CHECKPOINT=$(find ${WORK_DIR} -name "epoch_*.pth" -type f | sort -V | tail -n 1)
if [ -z "${CHECKPOINT}" ]; then
    echo "No checkpoint found in ${WORK_DIR}"
    exit 1
fi
echo "Found checkpoint: ${CHECKPOINT}"

# Run inference to generate results
echo "Generating detection results..."
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT} \
    --eval bbox \
    --out ${WORK_DIR}/results.pkl \
    --eval-options jsonfile_prefix=${RESULTS_DIR}/results_nusc

# Check if generation succeeded
if [ ! -f "${RESULTS_DIR}/results_nusc.json" ]; then
    echo "Failed to generate detection results JSON file."
    exit 1
fi

echo "Detection results generated at: ${RESULTS_DIR}/results_nusc.json"

# Create a simple BEV visualization script
cat > visualize_bev.py << 'EOF'
#!/usr/bin/env python
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pyquaternion import Quaternion
from PIL import Image

# Configuration
results_file = os.path.join("${RESULTS_DIR}", "results_nusc.json")
output_dir = "${VIS_DIR}"
num_visualizations = 20  # Number of samples to visualize
score_threshold = 0.3  # Detection score threshold

# Load results
print(f"Loading results from {results_file}")
try:
    with open(results_file, 'r') as f:
        results = json.load(f)
    print(f"Loaded results with {len(results['results'])} samples")
except Exception as e:
    print(f"Error loading results: {e}")
    exit(1)

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Process samples
sample_tokens = list(results['results'].keys())
if len(sample_tokens) == 0:
    print("No samples found in results")
    exit(1)

# Limit to specified number of visualizations
sample_tokens = sample_tokens[:num_visualizations]

# For each sample, create a simple visualization
for i, token in enumerate(sample_tokens):
    # Get detections for this sample
    dets = results['results'][token]
    
    # Filter by score
    dets = [d for d in dets if d['detection_score'] > score_threshold]
    
    if not dets:
        print(f"No detections above threshold for sample {token}")
        continue
    
    # Create a simple top-down visualization
    plt.figure(figsize=(10, 10))
    
    # Top-down view (Bird's eye view)
    plt.subplot(1, 1, 1)
    
    # Set up the plot as a top-down view
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.title(f"Bird's Eye View - Sample {token}")
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True)
    
    # Draw ego vehicle
    ego_circle = plt.Circle((0, 0), 1.0, color='blue', fill=True, alpha=0.7)
    plt.gca().add_patch(ego_circle)
    plt.text(0, 0, "EGO", ha='center', va='center', color='white', fontweight='bold')
    
    # Plot each detection
    colors = {
        'car': 'red',
        'truck': 'orange',
        'bus': 'green',
        'trailer': 'cyan',
        'construction_vehicle': 'brown',
        'pedestrian': 'magenta',
        'motorcycle': 'purple',
        'bicycle': 'pink',
        'traffic_cone': 'gray',
        'barrier': 'olive'
    }
    
    # Count detections by class
    class_counts = {}
    
    for det in dets:
        # Get coordinates
        x, y = det['translation'][0], det['translation'][1]
        w, l = det['size'][0], det['size'][1]  # width and length
        
        # Get rotation and convert to 2D rotation for plotting
        yaw = Quaternion(det['rotation']).yaw_pitch_roll[0]
        
        # Get detection class and color
        det_class = det['detection_name']
        color = colors.get(det_class, 'gray')
        
        # Update class count
        class_counts[det_class] = class_counts.get(det_class, 0) + 1
        
        # Create rectangle patch for vehicle
        rect = plt.Rectangle(
            (x - l/2, y - w/2),
            l, w,
            angle=np.degrees(yaw),
            linewidth=1,
            edgecolor=color,
            facecolor=color,
            alpha=0.3
        )
        plt.gca().add_patch(rect)
        
        # Add text for class and score
        plt.text(x, y, f"{det_class[:3]}\n{det['detection_score']:.2f}", 
                 ha='center', va='center', color='black', fontsize=8,
                 fontweight='bold')
    
    # Add legend with counts
    legend_elements = []
    for cls, count in class_counts.items():
        legend_elements.append(plt.Line2D([0], [0], color=colors.get(cls, 'gray'), 
                                        lw=0, marker='s', markersize=10,
                                        label=f'{cls}: {count}'))
    
    plt.legend(handles=legend_elements, loc='upper right', title='Objects')
    
    # Save the figure
    output_path = os.path.join(output_dir, f"{token}_bev.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Processed sample {i+1}/{len(sample_tokens)}: {token}")

print(f"Visualization complete! {len(sample_tokens)} samples visualized.")
print(f"Output directory: {output_dir}")
EOF

# Run the visualization script
echo "Running visualization script..."
python visualize_bev.py

echo "Process complete!" 