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

# CRITICAL: Use absolute path
cd /project2/ywang234_1595/petr_v2/ayushgoy/PETR/

# Add mmdetection3d to Python path
export PYTHONPATH=/project2/ywang234_1595/petr_v2/ayushgoy/PETR/mmdetection3d:$PYTHONPATH

# Configuration
WORK_DIR="/scratch1/ayushgoy/work_dir/4frame_adaptive"
CONFIG_FILE="projects/configs/petrv2/petrv2_3frame_adaptive.py"
RESULTS_DIR="${WORK_DIR}/results_eval/pts_bbox"
VIS_DIR="${WORK_DIR}/visualizations/detection_results"

# Create necessary directories
mkdir -p ${RESULTS_DIR}
mkdir -p ${VIS_DIR}

# Check if mmdet3d is installed correctly
python -c "
try:
    import mmdet3d
    print('mmdet3d import successful, version:', mmdet3d.__version__)
except ImportError as e:
    print('ERROR: mmdet3d import failed:', e)
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
