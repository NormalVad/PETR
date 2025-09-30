#!/bin/bash
#SBATCH --account=ywang234_1595
#SBATCH --partition=gpu
#SBATCH --nodes=4                    # 4 nodes × 2 GPUs/node = 8 GPUs total
#SBATCH --ntasks=8                   # one task per GPU
#SBATCH --ntasks-per-node=2          # 2 tasks (GPUs) on each node
#SBATCH --cpus-per-task=8            # e.g. 8 CPUs per task
#SBATCH --gres=gpu:2                 # request 2 GPUs per node
#SBATCH --mem=48G                    # Memory per node (replacing mem-per-gpu)
#SBATCH --time=48:00:00              # hh:mm:ss
#SBATCH --job-name=petr3d_3frame     # job name
#SBATCH --output=slurm_logs/%j.out   # %j will be replaced with the job ID
#SBATCH --error=slurm_logs/%j.err    # Separate error log

# Create log directory
mkdir -p slurm_logs

# Environment setup
source /home1/ayushgoy/.bashrc
conda deactivate
conda deactivate

module purge
module load conda
module load legacy/CentOS7  
module load gcc/9.2.0
module load cuda/11.2.2
export LD_LIBRARY_PATH=~/.conda/envs/677_project/lib:$LD_LIBRARY_PATH 
eval "$(conda shell.bash hook)"
conda activate 677_project

# Change to PETR directory
cd /project2/ywang234_1595/petr_v2/ayushgoy/PETR/

# Set environment variables
export GPUS=8
export GPUS_PER_NODE=2
export CPUS_PER_TASK=8

# Create output directory
WORK_DIR="/scratch1/ayushgoy/work_dir/3frame_adaptive"
DUMMY_STATS_DIR="${WORK_DIR}/dummy_stats"
mkdir -p ${DUMMY_STATS_DIR}

# Run training with 3-frame adaptive configuration
bash tools/slurm_train.sh gpu petr3d_3frame \
     projects/configs/petrv2/petrv2_3frame_adaptive.py \
     ${WORK_DIR} \
     --seed 42 \
     --cfg-options \
         data.train.ann_file='/scratch1/ayushgoy/nuscenes_extracted/mmdet3d_nuscenes_30f_infos_train.pkl' \
         data.val.ann_file='/scratch1/ayushgoy/nuscenes_extracted/mmdet3d_nuscenes_30f_infos_val.pkl' \
         data.test.ann_file='/scratch1/ayushgoy/nuscenes_extracted/mmdet3d_nuscenes_30f_infos_val.pkl' \
         data.train.save_interval=5000  # Save dummy stats every 5000 samples

# Wait for training to complete
wait

# Create visualization directories
VIS_RESULTS_DIR="${WORK_DIR}/visualizations/results"
ADAPTIVE_WEIGHTS_DIR="${WORK_DIR}/visualizations/adaptive_weights"
mkdir -p ${VIS_RESULTS_DIR}
mkdir -p ${ADAPTIVE_WEIGHTS_DIR}

echo "Training completed. Starting visualization..."

# Run inference to get results if not already generated
RESULTS_FILE="${WORK_DIR}/results.pkl"
CONFIG_FILE="projects/configs/petrv2/petrv2_3frame_adaptive.py"
CHECKPOINT_FILE="${WORK_DIR}/latest.pth"

# Check if checkpoint exists
if [ ! -f "${CHECKPOINT_FILE}" ]; then
    echo "Warning: Checkpoint file not found at ${CHECKPOINT_FILE}"
    # Try to find the most recent checkpoint
    LATEST_EPOCH=$(ls -v ${WORK_DIR}/epoch_*.pth 2>/dev/null | tail -n 1)
    if [ -n "${LATEST_EPOCH}" ]; then
        echo "Using most recent epoch checkpoint: ${LATEST_EPOCH}"
        CHECKPOINT_FILE="${LATEST_EPOCH}"
    else
        echo "Error: No checkpoint files found in ${WORK_DIR}. Skipping visualization."
        exit 1
    fi
fi

# Check if results file exists
if [ ! -f "${RESULTS_FILE}" ]; then
    echo "Results file not found. Running inference..."
    # Run inference to generate results
    python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} \
        --eval bbox \
        --show-dir ${WORK_DIR}/test_visuals \
        --cfg-options \
            data.test.ann_file='/scratch1/ayushgoy/nuscenes_extracted/mmdet3d_nuscenes_30f_infos_val.pkl' \
        --out ${RESULTS_FILE}
fi

# Verify results file was created successfully
if [ ! -f "${RESULTS_FILE}" ]; then
    echo "Error: Failed to generate results file. Skipping results visualization."
else
    # Visualize 3D detection results
    echo "Visualizing 3D detection results..."
    python tools/misc/visualize_3frame_results.py \
        ${CONFIG_FILE} \
        --checkpoint ${CHECKPOINT_FILE} \
        --result ${RESULTS_FILE} \
        --show-dir ${VIS_RESULTS_DIR} \
        --frames 3 \
        --score-thr 0.4
fi

# Visualize adaptive weights
echo "Visualizing adaptive weights..."
python tools/misc/visualize_adaptive_weights.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    --show-dir ${ADAPTIVE_WEIGHTS_DIR} \
    --frames 3 \
    --num-samples 5

echo "Visualization completed. Results saved to:"
echo "- 3D Detection Results: ${VIS_RESULTS_DIR}"
echo "- Adaptive Weights: ${ADAPTIVE_WEIGHTS_DIR}"
