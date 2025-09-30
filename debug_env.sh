#!/bin/bash
# Debug script to find and fix mmdet3d import issues
set -e  # Exit on error

echo "====================== ENVIRONMENT DEBUGGING ======================"
echo "Current directory: $(pwd)"

# Source bashrc and activate conda
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

echo "====================== CONDA ENVIRONMENT INFO ======================"
conda info
echo ""
echo "====================== PYTHON VERSION ======================"
python --version
echo ""
echo "====================== PYTHON PATH ======================"
python -c "import sys; print('\\n'.join(sys.path))"
echo ""
echo "====================== CHECKING MMDET3D INSTALLATION ======================"
echo "Conda install status:"
conda list | grep -E "mmdet|mmcv"
echo ""
echo "Pip install status:"
pip list | grep -E "mmdet|mmcv"
echo ""

echo "====================== FINDING MMDET3D MODULE ======================"
find ~/.conda/envs/677_project -name "mmdet3d" -type d
echo ""

echo "====================== ATTEMPTING IMPORT ======================"
python -c "
try:
    import mmdet
    print('mmdet import successful, version:', mmdet.__version__)
except ImportError as e:
    print('mmdet import failed:', e)

try:
    import mmdet3d
    print('mmdet3d import successful, version:', mmdet3d.__version__)
except ImportError as e:
    print('mmdet3d import failed:', e)

try:
    import mmcv
    print('mmcv import successful, version:', mmcv.__version__)
except ImportError as e:
    print('mmcv import failed:', e)
"

echo "====================== LOCATING PROJECT MMDET3D ======================"
cd /project2/ywang234_1595/petr_v2/ayushgoy/PETR/
find . -name "mmdet3d" -type d | head -5
echo ""

echo "====================== ATTEMPTING IMPORT FROM PROJECT DIR ======================"
python -c "
import sys
sys.path.insert(0, '/project2/ywang234_1595/petr_v2/ayushgoy/PETR/')
sys.path.insert(0, '/project2/ywang234_1595/petr_v2/ayushgoy/PETR/mmdetection3d/')

try:
    import mmdet
    print('mmdet import successful, version:', mmdet.__version__)
except ImportError as e:
    print('mmdet import failed:', e)

try:
    import mmdet3d
    print('mmdet3d import successful, version:', mmdet3d.__version__)
except ImportError as e:
    print('mmdet3d import failed:', e)
"

echo "====================== FIXING THE ISSUE ======================"
echo "Creating installation script..."

cat > fix_mmdet3d.sh << 'EOF'
#!/bin/bash
# Script to fix mmdet3d installation issues

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

# Go to PETR directory
cd /project2/ywang234_1595/petr_v2/ayushgoy/PETR/

# Check if mmdetection3d exists
if [ ! -d "mmdetection3d" ]; then
    echo "mmdetection3d directory not found. Cloning it..."
    git clone https://github.com/open-mmlab/mmdetection3d.git mmdetection3d
    cd mmdetection3d
    git checkout v0.17.1  # Use the version compatible with PETR
else
    cd mmdetection3d
    echo "mmdetection3d directory found at $(pwd)"
fi

# Install mmdetection3d in development mode
echo "Installing mmdetection3d in development mode..."
pip install -e .

# Go back to PETR directory
cd ..

# Now create a test script to verify imports
cat > test_imports.py << 'EOFPY'
import sys
print("Python path:")
for p in sys.path:
    print("  -", p)

print("\nTrying imports:")
import mmdet
print("mmdet:", mmdet.__version__)
import mmcv
print("mmcv:", mmcv.__version__)
import mmdet3d
print("mmdet3d:", mmdet3d.__version__)
print("\nAll imports successful!")
EOFPY

# Try running the test
echo "Running import test..."
python test_imports.py

echo "Now run your detection script with:"
echo "cd /project2/ywang234_1595/petr_v2/ayushgoy/PETR/"
echo "python tools/test.py projects/configs/petrv2/petrv2_3frame_adaptive.py /scratch1/ayushgoy/work_dir/4frame_adaptive/epoch_24.pth --eval bbox --out /scratch1/ayushgoy/work_dir/4frame_adaptive/results.pkl --eval-options jsonfile_prefix=/scratch1/ayushgoy/work_dir/4frame_adaptive/results_eval/pts_bbox/results_nusc"
EOF

chmod +x fix_mmdet3d.sh
echo "To fix the mmdet3d import issue, run: ./fix_mmdet3d.sh"

echo "====================== CREATING NEW DETECTION SCRIPT ======================"
cat > run_detections_fixed.sh << 'EOF'
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
EOF

chmod +x run_detections_fixed.sh
echo "Created a new detection script: run_detections_fixed.sh"

echo "====================== DEBUGGING COMPLETE ======================" 