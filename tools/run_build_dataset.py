import os
import sys

# Force the correct data root path
os.environ['NUSCENES_DATAROOT'] = '/scratch1/ayushgoy/nuscenes_extracted'

# Execute the original script with the correct path context
sys.path.insert(0, os.path.abspath('.'))
original_script = os.path.join('tools', 'build-dataset.py')
print(f"Running {original_script} with DATA_ROOT={os.environ['NUSCENES_DATAROOT']}")
exec(open(original_script).read())
