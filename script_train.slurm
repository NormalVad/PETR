#!/bin/bash
#SBATCH --account=ywang234_1595
#SBATCH --partition=gpu
#SBATCH --nodes=4                    # 4 nodes × 2 GPUs/node = 8 GPUs total
#SBATCH --ntasks=8                   # one task per GPU
#SBATCH --ntasks-per-node=2          # 2 tasks (GPUs) on each node
#SBATCH --cpus-per-task=8            # e.g. 8 CPUs per task
#SBATCH --gres=gpu:2                 # request 2 GPUs per node
#SBATCH --mem-per-gpu=24G            # 24 GB RAM per GPU
#SBATCH --time=48:00:00              # hh:mm:ss        


source /home1/ayushgoy/.bashrc
conda deactivate
conda deactivate

# module load gcc/11.3.0

# module purge
# module load conda
# module load legacy/CentOS7  
# module load gcc/9.2.0  
# module load cuda/11.2.2
eval "$(conda shell.bash hook)"
conda activate 677_project

cd /project2/ywang234_1595/petr_v2/ayushgoy/PETR/

# override defaults in slurm_train.sh
export GPUS=8
export GPUS_PER_NODE=2
export CPUS_PER_TASK=8

# PARTITION, JOB_NAME, CONFIG, WORK_DIR, then any extra PY_ARGS (like --seed)
bash tools/slurm_train.sh gpu petr_v2_job \
     projects/configs/petrv2/petrv2_vovnet_gridmask_p4_800x320.py \
     /scratch1/ayushgoy/work_dir/ \
     --seed 0

# tools/dist_train.sh projects/configs/petrv2/petrv2_vovnet_gridmask_p4_800x320.py 8 --work-dir /project2/ywang234_1595/petr_v2/PETRv2/ \
    # --launcher="slurm" --seed 0

# srun tools/dist_train.sh projects/configs/petrv2/petrv2_vovnet_gridmask_p4_800x320.py 8 \
#     --work-dir /scratch1/lbhatnag/petr_v2/slurm_train/ --seed 0

# srun tools/dist_train.sh projects/configs/petrv2/petrv2_vovnet_gridmask_p4_800x320.py 1  --work-dir /scratch1/ayushgoy/work_dir --seed 0
