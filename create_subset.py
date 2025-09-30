import os
import random # Keep for potential future use, though selection is commented
import json
import shutil
import subprocess
import time
import pickle # Keep for potential future use
from concurrent.futures import ThreadPoolExecutor, as_completed
import fnmatch # For checking map files

# Try importing necessary libraries and provide instructions if missing
try:
    from nuscenes.nuscenes import NuScenes
except ImportError:
    print("Error: nuscenes-devkit not found.")
    print("Please install it: pip install nuscenes-devkit")
    exit()

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not found. Progress bars will not be shown.")
    print("Install it for a better experience: pip install tqdm")
    # Define a dummy tqdm if not installed
    def tqdm(iterable, *args, **kwargs):
        yield from iterable # Use yield from for Python 3.3+ compatibility

# ==============================================================================
# Configuration
# ==============================================================================

# --- Define Paths (VERIFY THESE ARE CORRECT FOR YOUR SETUP) ---
# Directory where the FULL dataset IS ALREADY EXTRACTED
extracted_dir = '/scratch1/ayushgoy/nuscenes_extracted'
# Final destination directory FOR THE SUBSET (already partially populated)
subset_dir = '/project2/ywang234_1595/petr_v2/nuscenes_subset'
# Optional: Path to the PETR project (not strictly needed for this task)
# petr_dir = '/project2/ywang234_1595/petr_v2/ayushgoy/PETR/'


# --- Subset Configuration ---
# Define path to your EXISTING scene token list
subset_scenes_filepath = os.path.join(subset_dir, 'subset_scenes.json')

# --- Parallelism Configuration ---
# Set a specific limit for worker threads to avoid system limits
# Adjust this value if you still encounter thread errors or based on cluster recommendations
MAX_WORKERS = 32 # More conservative limit
print(f"Using a maximum of {MAX_WORKERS} workers for parallel operations.")


# ==============================================================================
# Helper Function for Parallel Copying (Keep this)
# ==============================================================================

def copy_single_file(args):
    """Copies a single file from source root to destination root.
       Assumes destination existence check was done before calling."""
    relative_filename, source_root, dest_root = args
    source_file = os.path.join(source_root, relative_filename)
    dest_file = os.path.join(dest_root, relative_filename)
    try:
        # Ensure the destination subdirectory exists (thread-safe)
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)

        # Check source exists *just before* copying (though unlikely to vanish)
        if not os.path.exists(source_file):
             return False, f"Source suddenly not found: {source_file}"

        # Perform the copy
        shutil.copy2(source_file, dest_file) # copy2 preserves metadata
        return True, relative_filename # Indicate success

    except Exception as e:
        return False, f"Error copying {relative_filename}: {e}" # Indicate failure

# ==============================================================================
# Step 1: Extraction (COMMENTED OUT - Assumed already done)
# ==============================================================================
# print("\n--- Step 1: Extracting Dataset Files ---")
# ... (Extraction logic commented out) ...
# print("Skipping extraction phase.")


# ==============================================================================
# Step 2: Initialize NuScenes & Load Subset Scene Info (Keep parts of this)
# ==============================================================================
print("\n--- Step 2: Initializing NuScenes & Loading Subset Scene Info ---")

# --- Initialize NuScenes Devkit (Needed to map sweeps to scenes) ---
print(f"Initializing NuScenes from FULL extracted dataset: {extracted_dir}")
start_time_init = time.time()
try:
    # Make sure this points to the FULL dataset root where metadata JSONs are
    nusc = NuScenes(version='v1.0-trainval', dataroot=extracted_dir, verbose=False)
    print(f"NuScenes initialized in {time.time() - start_time_init:.2f} seconds.")
except Exception as e:
    print(f"Error initializing NuScenes: {e}")
    print(f"Please ensure the full dataset (including metadata JSONs like scene.json, sample.json, sample_data.json)")
    print(f"is correctly located within: {extracted_dir}")
    exit()

# --- Load Selected Scene Tokens from existing JSON ---
print(f"Loading selected scene tokens from: {subset_scenes_filepath}")
if not os.path.exists(subset_scenes_filepath):
    print(f"Error: Cannot find the scene token list file: {subset_scenes_filepath}")
    print("This file is required to know which sweeps to copy.")
    exit()

try:
    with open(subset_scenes_filepath, 'r') as f:
        subset_info = json.load(f)
        selected_scene_tokens = set(subset_info['scene_tokens']) # Use a set for fast lookups
    print(f"Loaded {len(selected_scene_tokens)} scene tokens for the existing subset.")
except Exception as e:
    print(f"Error reading or parsing {subset_scenes_filepath}: {e}")
    exit()

# --- Scene Selection Logic (COMMENTED OUT - using loaded tokens) ---
# ... (Scene selection commented out) ...
# --- Saving scene tokens (COMMENTED OUT - already exists) ---
# ... (Saving commented out) ...


# ==============================================================================
# Step 3: Identify SWEEP Files Belonging to Selected Scenes (Keep this)
# ==============================================================================
print("\n--- Step 3: Identifying Required SWEEP Files for Subset ---")
all_required_sweep_files = [] # List to store relative paths
start_time_identify = time.time()

print(f"Scanning {len(nusc.sample_data)} sample_data records to find relevant sweeps...")
# Use nusc.sample_data as the source of truth for file paths
for sd_record in tqdm(nusc.sample_data, desc="Identifying sweeps"):
    # Check if it's a sweep first (usually faster)
    if not sd_record['is_key_frame'] and 'sweeps/' in sd_record['filename']:
        try:
            sample_record = nusc.get('sample', sd_record['sample_token'])
            # Check if the sweep belongs to one of the selected scenes
            if sample_record['scene_token'] in selected_scene_tokens:
                all_required_sweep_files.append(sd_record['filename'])
        except KeyError:
            # This can happen if sample_token is invalid, dataset corruption?
            print(f"Warning: Could not find sample record for sample_data token {sd_record['token']}")

print(f"Identified {len(all_required_sweep_files)} total sweep files required for the {len(selected_scene_tokens)} selected scenes.")
print(f"Sweep identification took {time.time() - start_time_identify:.2f} seconds.")


# ==============================================================================
# Step 4: Copy Identified SWEEP Files in Parallel (with Resumability)
# ==============================================================================
print("\n--- Step 4: Copying Subset SWEEP Files (Resumable) ---")

# --- Filter out files that already exist in the destination ---
print("Checking for sweep files already present in the destination...")
sweeps_to_actually_copy = []
for relative_filename in tqdm(all_required_sweep_files, desc="Checking existing sweeps"):
    dest_file = os.path.join(subset_dir, relative_filename)
    if not os.path.exists(dest_file):
        sweeps_to_actually_copy.append(relative_filename)

if not sweeps_to_actually_copy:
    print("All required sweep files seem to be present already.")
else:
    print(f"Found {len(sweeps_to_actually_copy)} sweep files to copy (out of {len(all_required_sweep_files)} total required).")

    # --- Ensure Sweeps Directory Exists ---
    sweeps_target_dir = os.path.join(subset_dir, 'sweeps')
    os.makedirs(sweeps_target_dir, exist_ok=True)

    # --- Copy Sweep Files (KEEP THIS) ---
    print(f"\nCopying {len(sweeps_to_actually_copy)} remaining sweep files using max {MAX_WORKERS} workers...")
    copied_sweep_count = 0
    failed_sweep_copies = []
    start_time_copy_sweeps = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
         # Create argument list ONLY for files that need copying
        tasks = [(fname, extracted_dir, subset_dir) for fname in sweeps_to_actually_copy]
        futures = {executor.submit(copy_single_file, task): task[0] for task in tasks}

        for future in tqdm(as_completed(futures), total=len(sweeps_to_actually_copy), desc="Copying Sweeps"):
            filename = futures[future] # Get filename associated with the completed future
            try:
                success, message = future.result()
                if success:
                    copied_sweep_count += 1
                else:
                    failed_sweep_copies.append(message)
            except Exception as exc:
                failed_sweep_copies.append(f"Task for sweep {filename} generated exception: {exc}")

    print(f"\nFinished sweep copy attempt in {time.time() - start_time_copy_sweeps:.2f} seconds.")
    print(f"Successfully copied {copied_sweep_count} sweep files in this run.")
    if failed_sweep_copies:
        print(f"Encountered {len(failed_sweep_copies)} errors/warnings during sweep copy:")
        # Print first few errors for diagnosis
        for i, fail_msg in enumerate(failed_sweep_copies[:10]):
            print(f"  - {fail_msg}")
        if len(failed_sweep_copies) > 10:
            print(f"  ... and {len(failed_sweep_copies) - 10} more.")


# ==============================================================================
# Step 5: Copy Map Files (with Resumability Check)
# ==============================================================================
print("\n--- Step 5: Copying Map Files (Resumable) ---")

map_source_path = os.path.join(extracted_dir, 'maps')
map_dest_path = os.path.join(subset_dir, 'maps')

if not os.path.exists(map_source_path):
     print(f"Warning: Source map directory not found in full extraction: {map_source_path}. Cannot copy maps.")
else:
    # Check if maps seem reasonably complete in destination
    # Example check: look for a few key files/folders expected from the expansion
    required_map_elements = [
        'basemap/boston-seaport.json',
        'expansion/boston-seaport_corr.json',
        'prediction/prediction_scenes.json' # Add more if needed
    ]
    maps_seem_complete = True
    for element in required_map_elements:
        if not os.path.exists(os.path.join(map_dest_path, element)):
            maps_seem_complete = False
            print(f"  Map element missing in destination: {element}")
            break

    if maps_seem_complete:
        print("Maps directory seems populated in the subset. Skipping map copy.")
    else:
        print(f"Copying maps from {map_source_path} to {map_dest_path}...")
        start_time_map_copy = time.time()
        try:
            # Ensure target map directory exists before copytree
            os.makedirs(map_dest_path, exist_ok=True)
            # Use shutil.copytree for simplicity. `dirs_exist_ok=True` handles if target exists.
            # Note: copytree doesn't easily skip existing files like rsync.
            # For true resumability here, rsync is better.
            print("Attempting copy using rsync (recommended for maps)...")
            try:
                # Ensure trailing slash on source for rsync to copy contents
                # -a: archive (recursive, preserves perms, etc.)
                # -q: quiet (can remove for more verbose output)
                # --ignore-existing: Skip files that already exist in the destination
                rsync_command = ['rsync', '-aq', '--ignore-existing', f"{map_source_path.rstrip('/')}/", map_dest_path]
                print(f"  Executing: {' '.join(rsync_command)}")
                subprocess.run(rsync_command, check=True)
                print(f"Map files potentially updated/copied using rsync in {time.time() - start_time_map_copy:.2f} seconds.")
            except Exception as rsync_e:
                print(f"rsync failed: {rsync_e}.")
                print("Falling back to simple copytree (may recopy existing files)...")
                shutil.copytree(map_source_path, map_dest_path, dirs_exist_ok=True)
                print(f"Map files copied using copytree in {time.time() - start_time_map_copy:.2f} seconds (might have recopied).")

        except Exception as e:
            print(f"Error during map copy: {e}. Please copy maps manually.")
            print(f"Command: rsync -av --ignore-existing {map_source_path}/ {map_dest_path}/")


# ==============================================================================
# Step 6: Create Subset Info File (.pkl) (COMMENTED OUT - Assumed exists)
# ==============================================================================
# print("\n--- Step 6: Creating Subset Info File ---")
# ... (Info file logic commented out) ...


# ==============================================================================
# Final Summary
# ==============================================================================
print("\n--- Sweeps & Maps Population Summary ---")
print(f"Target Subset Directory: {subset_dir}")
print(f"Scene Tokens Loaded: {len(selected_scene_tokens)} (from {subset_scenes_filepath})")
print(f"Required Sweep Files: {len(all_required_sweep_files)}")
print(f"Sweep Files Copied in this run: {copied_sweep_count} (approx)")
print(f"Maps Copied/Checked: Yes (see messages above for status)")

print("\n--- Next Steps ---")
print("1. Verify the contents of the subset directory, especially sweeps and maps:")
print(f"   ls -l {os.path.join(subset_dir, 'sweeps')}")
print(f"   ls -l {os.path.join(subset_dir, 'maps')}")
print("\n2. Ensure your PETR configuration file (.py) correctly points to the subset:")
print(f"   data_root = '{subset_dir}'")
print(f"   ann_file = data_root + 'YOUR_SUBSET_INFO_FILE.pkl' # Make sure this filename is correct")

print("\nScript finished focusing on sweeps and maps.")