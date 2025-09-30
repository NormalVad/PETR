#!/bin/bash
# Script to run visualization for 4-frame PETR results
set -e  # Exit on error

# Create visualization directories
WORK_DIR="/scratch1/ayushgoy/work_dir/4frame_adaptive"
RESULTS_DIR="${WORK_DIR}/results_eval/pts_bbox"
VIS_DIR="${WORK_DIR}/visualizations"

mkdir -p ${VIS_DIR}
mkdir -p ${VIS_DIR}/detection_results
mkdir -p ${VIS_DIR}/weights

# Ensure the Python environment is properly set up
source /home1/ayushgoy/.bashrc
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true

# Load modules
module purge
module load conda
module load legacy/CentOS7
module load gcc/9.2.0
module load cuda/11.2.2

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate 677_project

# Make sure Python can find the necessary modules
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Check if results file exists
if [ ! -d "${RESULTS_DIR}" ]; then
    # If results directory doesn't exist, create it and generate results
    mkdir -p ${RESULTS_DIR}
    
    # Find the latest checkpoint
    CHECKPOINT=$(find ${WORK_DIR} -name "epoch_*.pth" -type f | sort -V | tail -n 1)
    
    if [ -z "${CHECKPOINT}" ]; then
        echo "No checkpoint found in ${WORK_DIR}"
        exit 1
    fi
    
    echo "Found checkpoint: ${CHECKPOINT}"
    echo "Generating results using this checkpoint..."
    
    # Run inference to generate results
    python tools/test.py projects/configs/petrv2/petrv2_3frame_adaptive.py ${CHECKPOINT} \
        --eval bbox \
        --out ${WORK_DIR}/results.pkl \
        --eval-options jsonfile_prefix=${RESULTS_DIR}/results_nusc
fi

# Run visualization
echo "Running 4-frame visualization script..."
python visualize_petr_4frame.py

echo "Visualization complete! Results saved to: ${VIS_DIR}"
echo "- Detection results: ${VIS_DIR}/detection_results"
echo "- Adaptive weights: ${VIS_DIR}/weights" 