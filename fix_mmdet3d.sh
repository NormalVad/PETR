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
