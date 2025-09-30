#!/bin/bash
#SBATCH --account=ywang234_1595
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=petr_filter_gt

# ----------------- Path Configuration -----------------
DATA_ROOT="/scratch1/ayushgoy/nuscenes_extracted"
PROJECT_DIR="/project2/ywang234_1595/petr_v2/ayushgoy/PETR"
SUBSET_JSON="${PROJECT_DIR}/nuscenes_subset/subset_scenes.json"

# ----------------- Diagnostic Setup -----------------
function debug_paths() {
    echo "=== PATH DIAGNOSTICS ==="
    echo "Host: $(hostname)"
    echo "Date: $(date)"
    echo "CWD : $(pwd)"
    echo "DATA_ROOT: $(ls -ld ${DATA_ROOT} 2>/dev/null || echo MISSING)"
    echo "PROJECT_DIR: $(ls -l ${PROJECT_DIR}/ 2>/dev/null || echo UNAVAILABLE)"
    echo "SCRATCH1: $(df -h /scratch1 | tail -1)"
}

debug_paths

# ----------------- Environment Setup -----------------
eval "$(conda shell.bash hook)"
conda activate 677_project || { 
    echo "ERROR: Conda activation failed"
    exit 1
}

# ----------------- Filter Execution -----------------
cd "${PROJECT_DIR}" || exit 1

# Run filter with explicit paths
# python nuscenes_subset/filter_pkl.py \
#     --nusc-root "${DATA_ROOT}" \
#     --subset-json "${SUBSET_JSON}" \
#     --full-train "${DATA_ROOT}/nuscenes_infos_train.pkl" \
#     --full-val "${DATA_ROOT}/nuscenes_infos_val.pkl" \
#     --out-train "${DATA_ROOT}/nuscenes_infos_train_subset.pkl" \
#     --out-val "${DATA_ROOT}/nuscenes_infos_val_subset.pkl"

# Verify outputs
# echo "=== Filtered Files ==="
# ls -lh "${DATA_ROOT}"/*_subset.pkl

# ----------------- Generate Sweep Data -----------------
# echo "=== Generating Sweep Data ==="
# python "${PROJECT_DIR}/tools/generate_sweep_pkl.py" \
#     --data-root "${DATA_ROOT}" \
#     --info-prefix "train" \
#     --num-prev 5 \
#     --num-sweep 5 \
#     --input-pkl "${DATA_ROOT}/nuscenes_infos_train_subset.pkl" \
#     --output-pkl "${DATA_ROOT}/nuscenes_infos_train_sweep.pkl"

# ----------------- Ground Truth Database Creation -----------------
echo "=== CREATING GT DATABASE ==="
cd "${PROJECT_DIR}/tools/data_converter"

python create_gt_database.py \
    --dataset_name NuScenesDataset \
    --data_root "${DATA_ROOT}" \
    --info_prefix nuscenes \
    --info_path "${DATA_ROOT}/nuscenes_dataset_infos_train_subset.pkl" \
    --db_info_save_path "${DATA_ROOT}/nuscenes_subset_dbinfos_train.pkl" \
    --database_save_path "${DATA_ROOT}/gt_db"

# ----------------- Cleanup & Verification -----------------
echo "=== FINAL VERIFICATION ==="
ls -lh "${DATA_ROOT}"/nuscenes_*subset*.pkl
ls -ld "${DATA_ROOT}/gt_db"
echo "=== JOB COMPLETED SUCCESSFULLY ==="

# Remove temporary symlinks
rm -f "${DATA_ROOT}/nuscenes_dataset_infos_train.pkl" "${DATA_ROOT}/nuscenes_dataset_infos_val.pkl"