import os
import random
import json
import shutil
from nuscenes.nuscenes import NuScenes

# Initialize the full dataset
nusc = NuScenes(version='v1.0-trainval', dataroot='/path/to/nuscenes_full', verbose=True)

# Create directories for the subset
subset_dir = '/path/to/nuscenes_subset'
os.makedirs(subset_dir, exist_ok=True)

# Get all scenes
all_scenes = nusc.scene
print(f"Total scenes in dataset: {len(all_scenes)}")

# Randomly select 15% of the scenes
num_scenes_to_select = int(len(all_scenes) * 0.15)
selected_scenes = random.sample(all_scenes, num_scenes_to_select)
selected_scene_tokens = [scene['token'] for scene in selected_scenes]

print(f"Selected {num_scenes_to_select} scenes for the subset")

# Create a list to track all sample tokens in selected scenes
subset_sample_tokens = []
subset_sample_data_tokens = []

# Collect all sample tokens from selected scenes
for scene_token in selected_scene_tokens:
    scene = nusc.get('scene', scene_token)
    sample_token = scene['first_sample_token']
    
    # Traverse all samples in the scene
    while sample_token:
        subset_sample_tokens.append(sample_token)
        sample = nusc.get('sample', sample_token)
        
        # Collect all sensor data tokens for this sample
        for _, sd_token in sample['data'].items():
            subset_sample_data_tokens.append(sd_token)
        
        sample_token = sample['next']

# Copy the necessary files to create the subset
# 1. Copy metadata file structure (modified to include only subset data)
os.makedirs(os.path.join(subset_dir, 'maps'), exist_ok=True)
os.makedirs(os.path.join(subset_dir, 'samples'), exist_ok=True)
os.makedirs(os.path.join(subset_dir, 'sweeps'), exist_ok=True)

# 2. Copy actual sensor data files
for sd_token in subset_sample_data_tokens:
    sd_record = nusc.get('sample_data', sd_token)
    source_file = os.path.join(nusc.dataroot, sd_record['filename'])
    dest_file = os.path.join(subset_dir, sd_record['filename'])
    
    # Create directory structure if needed
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
    
    # Copy the file
    shutil.copy2(source_file, dest_file)
    
    # If this is a keyframe, also copy its corresponding annotations
    if sd_record['is_key_frame']:
        # Copy relevant annotation files if they exist
        sample = nusc.get('sample', sd_record['sample_token'])
        # ... copy annotation files as needed

# 3. Save the list of selected scene tokens for reference
with open(os.path.join(subset_dir, 'subset_scenes.json'), 'w') as f:
    json.dump({
        'scene_tokens': selected_scene_tokens,
        'total_scenes': len(all_scenes),
        'subset_percentage': 15
    }, f, indent=2)

print(f"Created subset with {num_scenes_to_select} scenes in {subset_dir}")
