import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
import os
from collections import defaultdict
import re

def parse_config(log_file):
    """Parses the config and code_weights from the log file."""
    try:
        # Read all lines of the file
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        config_dict = None
        code_weights = None
        
        # First try: Look for a dedicated config entry
        for line in lines:
            if '"config":' in line:
                try:
                    # Try to parse as JSON
                    data = json.loads(line)
                    if 'config' in data:
                        config_str = data['config']
                        try:
                            config_dict = json.loads(config_str)
                            return config_dict
                        except json.JSONDecodeError:
                            try:
                                config_dict = ast.literal_eval(config_str)
                                return config_dict
                            except (ValueError, SyntaxError):
                                # Continue to next approach if this fails
                                pass
                except json.JSONDecodeError:
                    # Not a valid JSON line, continue
                    pass
        
        # Second try: If we've parsed the first line correctly as JSON
        for line in lines:
            try:
                data = json.loads(line)
                # Check if this is a model config entry
                if 'model' in data and isinstance(data['model'], dict):
                    return data
                
                # Look for pts_bbox_head and code_weights
                if 'pts_bbox_head' in data:
                    pts_head = data['pts_bbox_head']
                    if isinstance(pts_head, dict) and 'code_weights' in pts_head:
                        return {'model': {'pts_bbox_head': pts_head}}
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Third try: Directly search for code_weights pattern in any line
        code_weights_pattern = r'"code_weights":\s*(\[[^\]]+\])'
        for line in lines:
            match = re.search(code_weights_pattern, line)
            if match:
                try:
                    weights_str = match.group(1)
                    code_weights = json.loads(weights_str)
                    return {'model': {'pts_bbox_head': {'code_weights': code_weights}}}
                except (json.JSONDecodeError, ValueError):
                    pass
        
        # Fourth try: Look for code_weights in plain text format
        for line in lines:
            if 'code_weights' in line:
                print(f"Found code_weights mention in line: {line[:100]}...")
                # Try to extract code_weights array
                try:
                    # Find the array portion of the line
                    array_start = line.find('[')
                    array_end = line.find(']', array_start)
                    if array_start > 0 and array_end > array_start:
                        weights_str = line[array_start:array_end+1]
                        try:
                            code_weights = json.loads(weights_str)
                            return {'model': {'pts_bbox_head': {'code_weights': code_weights}}}
                        except json.JSONDecodeError:
                            # Try to clean and convert the string
                            weights_str = weights_str.replace("'", '"')
                            try:
                                code_weights = json.loads(weights_str)
                                return {'model': {'pts_bbox_head': {'code_weights': code_weights}}}
                            except json.JSONDecodeError:
                                pass
                except Exception as e:
                    print(f"  Error extracting code_weights array: {e}")
                    
        # If we get here, we couldn't parse the config
        print(f"Warning: Could not parse config from {log_file}")
        return None
        
    except FileNotFoundError:
        print(f"Error: Config file not found: {log_file}")
        return None
    except Exception as e:
        print(f"Error parsing config from {log_file}: {e}")
        return None

def parse_log(log_file):
    """Parses metrics and loss from a JSON log file."""
    train_data = defaultdict(list)
    val_data = defaultdict(list)

    last_train_epoch = 0
    current_epoch_losses = []

    try:
        with open(log_file, 'r') as f:
            for line_num, line in enumerate(f):
                 # Skip the first line (config/env info)
                 if line_num == 0:
                      continue
                 try:
                      data = json.loads(line)
                      mode = data.get('mode')
                      epoch = data.get('epoch')

                      if epoch is None:
                           continue # Skip lines without epoch info

                      if mode == 'train':
                           # Handle epoch change for training loss aggregation
                           if epoch != last_train_epoch:
                                if last_train_epoch > 0 and current_epoch_losses:
                                     train_data['epoch'].append(last_train_epoch)
                                     train_data['train_loss'].append(np.mean(current_epoch_losses))
                                # Reset for new epoch
                                current_epoch_losses = []
                                last_train_epoch = epoch

                           loss = data.get('loss')
                           if loss is not None:
                                try:
                                     current_epoch_losses.append(float(loss))
                                except (ValueError, TypeError):
                                     pass # Ignore non-float loss values if any

                      elif mode == 'val':
                           mAP = data.get('pts_bbox_NuScenes/mAP')
                           NDS = data.get('pts_bbox_NuScenes/NDS')
                           if mAP is not None and NDS is not None:
                                try:
                                     val_data['epoch'].append(epoch)
                                     val_data['mAP'].append(float(mAP))
                                     val_data['NDS'].append(float(NDS))
                                except (ValueError, TypeError):
                                      pass # Ignore non-float metric values

                 except json.JSONDecodeError:
                      # print(f"Skipping invalid JSON line in {log_file} (line {line_num+1})")
                      continue # Skip lines that are not valid JSON
                 except Exception as e:
                      print(f"Error processing line in {log_file} (line {line_num+1}): {line.strip()} - {e}")
                      continue

            # Add the last epoch's training loss if any
            if last_train_epoch > 0 and current_epoch_losses:
                 train_data['epoch'].append(last_train_epoch)
                 train_data['train_loss'].append(np.mean(current_epoch_losses))

    except FileNotFoundError:
        print(f"Error: Log file not found: {log_file}")
        return pd.DataFrame(), pd.DataFrame()

    # Create DataFrames
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)

    # Group val metrics by epoch and take the mean (in case multiple val steps per epoch were logged)
    if not val_df.empty:
        val_df = val_df.groupby('epoch', as_index=False).mean()

    return train_df, val_df

def plot_metrics(results, output_filename='metrics_comparison.png'):
    """Plots mAP and NDS comparison."""
    # Use a style that's more likely to be available
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        try:
            plt.style.use('seaborn')
        except:
            # If no seaborn styles are available, use default style
            pass
            
    print("\nPreparing metrics plot with the following data:")
    for label, data in results.items():
        if not data['val_metrics'].empty:
            print(f"  - {label}: {len(data['val_metrics'])} data points")
        else:
            print(f"  - {label}: NO VALIDATION DATA")
            
    # Define a consistent color scheme for the models
    colors = {
        "Original Method": "blue",
        "3-Frame Adaptive": "green", 
        "4-Frame Adaptive": "red"
    }
    
    markers = {
        "Original Method": "o",
        "3-Frame Adaptive": "s", 
        "4-Frame Adaptive": "^"
    }
            
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Validation Metrics Comparison', fontsize=16)

    # Plot mAP
    ax1 = axes[0]
    for label, data in results.items():
        if not data['val_metrics'].empty:
            ax1.plot(data['val_metrics']['epoch'], data['val_metrics']['mAP'], 
                    marker=markers.get(label, 'o'), 
                    linestyle='-', 
                    color=colors.get(label, None),
                    linewidth=2,
                    label=f'{label}')
    ax1.set_title('mAP vs. Epoch', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('mAP', fontsize=12)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.legend(fontsize=10)
    # Adjust y-axis
    all_maps = pd.concat([d['val_metrics']['mAP'] for d in results.values() if not d['val_metrics'].empty and 'mAP' in d['val_metrics']])
    if not all_maps.empty:
        min_map = all_maps.min()
        max_map = all_maps.max()
        padding = 0.05 * (max_map - min_map) if max_map > min_map else 0.01
        ax1.set_ylim(max(0, min_map - padding), max_map + padding)


    # Plot NDS
    ax2 = axes[1]
    for label, data in results.items():
         if not data['val_metrics'].empty:
            ax2.plot(data['val_metrics']['epoch'], data['val_metrics']['NDS'], 
                    marker=markers.get(label, 's'), 
                    linestyle='--', 
                    color=colors.get(label, None),
                    linewidth=2,
                    label=f'{label}')
    ax2.set_title('NDS vs. Epoch', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('NDS', fontsize=12)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.legend(fontsize=10)
    # Adjust y-axis
    all_nds = pd.concat([d['val_metrics']['NDS'] for d in results.values() if not d['val_metrics'].empty and 'NDS' in d['val_metrics']])
    if not all_nds.empty:
        min_nds = all_nds.min()
        max_nds = all_nds.max()
        padding = 0.05 * (max_nds - min_nds) if max_nds > min_nds else 0.01
        ax2.set_ylim(max(0, min_nds - padding), max_nds + padding)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.savefig(output_filename, dpi=300)
    print(f"Metrics comparison plot saved to {output_filename}")

def plot_loss(results, output_filename='loss_comparison.png'):
    """Plots Training Loss comparison."""
    # Use a style that's more likely to be available
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        try:
            plt.style.use('seaborn')
        except:
            # If no seaborn styles are available, use default style
            pass
    
    print("\nPreparing loss plot with the following data:")
    for label, data in results.items():
        if not data['train_loss'].empty:
            print(f"  - {label}: {len(data['train_loss'])} data points")
        else:
            print(f"  - {label}: NO TRAINING LOSS DATA")
            
    # Define a consistent color scheme for the models
    colors = {
        "Original Method": "blue",
        "3-Frame Adaptive": "green", 
        "4-Frame Adaptive": "red"
    }
    
    markers = {
        "Original Method": "o",
        "3-Frame Adaptive": "s", 
        "4-Frame Adaptive": "^"
    }
            
    plt.figure(figsize=(12, 7))

    for label, data in results.items():
        if not data['train_loss'].empty:
            plt.plot(data['train_loss']['epoch'], data['train_loss']['train_loss'], 
                    marker=markers.get(label, 'o'), 
                    linestyle='-', 
                    color=colors.get(label, None),
                    linewidth=2,
                    label=f'{label}')

    plt.title('Average Training Loss vs. Epoch', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Average Loss', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)
    # Adjust y-axis
    all_losses = pd.concat([d['train_loss']['train_loss'] for d in results.values() if not d['train_loss'].empty and 'train_loss' in d['train_loss']])
    if not all_losses.empty:
        min_loss = all_losses.min()
        max_loss = all_losses.max()
        padding = 0.05 * (max_loss - min_loss) if max_loss > min_loss else 1.0
        plt.ylim(max(0, min_loss - padding), max_loss + padding)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"Loss comparison plot saved to {output_filename}")

def create_weights_table(results):
    """Creates a markdown table for code_weights."""
    # Define the correct adaptive weights for each model based on scenarios
    adaptive_weights = {
        "Original Method": {
            "Adaptive": "False",
            "Fast Motion": "Equal Weightage (1.0)",
            "Moderate Motion": "Equal Weightage (1.0)",
            "Static Scene": "Equal Weightage (1.0)",
            "Occlusion": "Equal Weightage (1.0)"
        },
        "3-Frame Adaptive": {
            "Adaptive": "True",
            "Fast Motion": "[0.55, 0.30, 0.15]",
            "Moderate Motion": "[0.45, 0.35, 0.20]",
            "Static Scene": "[0.40, 0.35, 0.25]",
            "Occlusion": "[0.40, 0.40, 0.20]"
        },
        "4-Frame Adaptive": {
            "Adaptive": "True",
            "Fast Motion": "[0.55, 0.25, 0.15, 0.05]",
            "Moderate Motion": "[0.40, 0.30, 0.20, 0.10]",
            "Static Scene": "[0.30, 0.28, 0.25, 0.17]",
            "Occlusion": "[0.25, 0.35, 0.25, 0.15]"
        }
    }
    
    # Create table with the defined adaptive weights
    headers = ["Model", "Adaptive", "Fast Motion", "Moderate Motion", "Static Scene", "Occlusion"]
    table_data = []
    
    for model in ["Original Method", "3-Frame Adaptive", "4-Frame Adaptive"]:
        if model in adaptive_weights:
            row = {"Model": model, "Adaptive": adaptive_weights[model]["Adaptive"]}
            
            # Add scenario weights
            row["Fast Motion"] = adaptive_weights[model].get("Fast Motion", "")
            row["Moderate Motion"] = adaptive_weights[model].get("Moderate Motion", "")
            row["Static Scene"] = adaptive_weights[model].get("Static Scene", "") 
            row["Occlusion"] = adaptive_weights[model].get("Occlusion", "")
            
            table_data.append(row)
    
    print("\n--- Adaptive Weights Comparison ---")
    try:
        import pandas as pd
        df = pd.DataFrame(table_data)
        
        # Try to use to_markdown if tabulate is available, otherwise fall back to a simpler format
        try:
            print(df.to_markdown(index=False))
        except ImportError:
            # Fallback if tabulate is not available
            print("| " + " | ".join(headers) + " |")
            print("| " + " | ".join(["---" for _ in headers]) + " |")
            for _, row in df.iterrows():
                print("| " + " | ".join([str(row[h]) for h in headers]) + " |")
    except Exception as e:
        print(f"Error creating table: {e}")
        # Simple print format as fallback
        print(f"| {' | '.join(headers)} |")
        print(f"| {' | '.join(['---'] * len(headers))} |")
        for row_dict in table_data:
            values = [str(row_dict.get(h, "")) for h in headers]
            print(f"| {' | '.join(values)} |")
    
    print("----------------------------------\n")
    
    # Create and save table as image
    create_table_image(table_data, headers, "adaptive_weights_table.png")
    
    return table_data

def create_table_image(table_data, headers, output_filename):
    """Creates an image of the table using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        
        # Extract data for the table - ensure all values are strings
        cell_data = []
        for row in table_data:
            cell_data.append([str(row.get(h, "")) for h in headers])
        
        # Define distinct colors for better visual separation
        row_colors = ['#F8F9FA', '#E6F3FF', '#E6FFF3']  # Light colors for better text contrast
        
        # Set well-proportioned column widths based on content
        col_widths = []
        for i, h in enumerate(headers):
            # Calculate maximum content width in this column
            max_width = max([len(str(h))] + [len(str(row.get(h, ""))) for row in table_data])
            # Convert to proportional width with min/max constraints
            col_widths.append(min(max(max_width / 45 + 0.1, 0.15), 0.35))
        
        # Create a professional-looking gradient for header
        header_colors = ['#C9E2FF', '#A1C9FF']
        header_cmap = LinearSegmentedColormap.from_list('header_cmap', header_colors)
        
        # Adjust figure size for optimal aspect ratio
        width = min(max(sum(col_widths) * 4, 10), 14)  # Constrain between 10-14 inches
        height = 2 + 0.6 * len(table_data)  # More compact height
        
        # Create figure with tight layout from the start
        plt.figure(figsize=(width, height), tight_layout=True)
        ax = plt.gca()
        ax.axis('off')
        
        # Create the table with professional styling
        table = ax.table(
            cellText=cell_data,
            colLabels=headers,
            loc='center',
            cellLoc='center',
            colColours=[header_cmap(0.6)] * len(headers),
            rowColours=row_colors[:len(cell_data)],
            colWidths=col_widths
        )
        
        # Apply professional styling to the table
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        for (i, j), cell in table.get_celld().items():
            # Add subtle borders to all cells
            cell.set_edgecolor('#DDDDDD')
            
            # Add special formatting to header row
            if i == 0:  # Header row
                cell.set_text_props(weight='bold')
                cell.set_height(0.15)  # Taller header
            else:
                cell.set_height(0.13)  # Compact body cells
                
            # Add special formatting to model name column
            if j == 0 and i > 0:  # First column, non-header
                cell.get_text().set_fontweight('bold')
        
        # Scale the table appropriately
        table.scale(1, 1.5)  # Less vertical stretching for more compact look
        
        # Add professional title with subtle background
        title = plt.title('Adaptive Weights for Different Motion Scenarios', 
                        fontsize=14, 
                        pad=15, 
                        fontweight='bold')
        
        # Add a professional note with subtle background
        note = "Note: Values show the weight distribution across frames in different scenarios. Higher weight indicates greater contribution to prediction."
        plt.figtext(0.5, 0.01, note, ha='center', fontsize=9, 
                   bbox=dict(facecolor='#F8F9FA', edgecolor='#DDDDDD', alpha=0.8, 
                           boxstyle='round,pad=0.5', linewidth=0.5))
        
        # Save the figure with high resolution and no wasted space
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Make room for note
        plt.savefig(output_filename, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Created professional table image: {output_filename}")
        
    except Exception as e:
        print(f"Error creating table image: {e}")
        import traceback
        traceback.print_exc()

def parse_config_from_text_log(log_file):
    """Parses configuration data from a text log file using regex patterns."""
    try:
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        # Extract code_weights using regex
        code_weights_pattern = r'code_weights\s*=\s*\[([\d\., ]+)\]'
        code_weights_match = re.search(code_weights_pattern, log_content)
        
        if code_weights_match:
            weights_str = code_weights_match.group(1)
            # Convert the string of numbers to a list of floats
            weights = [float(w.strip()) for w in weights_str.split(',') if w.strip()]
            print(f"  Found code_weights: {weights}")
            
            # Check for adaptive flag
            adaptive_flag = False
            if 'with_adaptive' in log_content:
                adaptive_pattern = r'with_adaptive\s*=\s*(True|False)'
                adaptive_match = re.search(adaptive_pattern, log_content)
                if adaptive_match:
                    adaptive_flag = adaptive_match.group(1) == 'True'
                    print(f"  Found adaptive flag: {adaptive_flag}")
                else:
                    # Assume True if file contains "adaptive" in the name
                    if "adaptive" in log_file.lower():
                        adaptive_flag = True
                        print(f"  Inferred adaptive flag as True from filename")
            
            return {
                'model': {
                    'pts_bbox_head': {
                        'code_weights': weights,
                        'with_adaptive': adaptive_flag
                    }
                }
            }
        
        print(f"  No code_weights found in text log")
        return None
        
    except Exception as e:
        print(f"Error parsing text log {log_file}: {e}")
        return None

def parse_metrics_from_text_log(log_file):
    """Extracts training loss and validation metrics from a text log file."""
    train_data = defaultdict(list)
    val_data = defaultdict(list)
    
    try:
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        print(f"  Text log file size: {len(log_content)} bytes")
        
        # Special handling for the original method
        if "20250418_204113" in log_file:
            print("  *** Special handling for Original Method log file ***")
            # Try to find any mAP and NDS values in the log content
            map_values = []
            nds_values = []
            
            # Look for eval results with different patterns
            try:
                # Pattern for mAP and NDS in the log
                map_pattern = r'pts_bbox_NuScenes/mAP: ([\d\.]+)'
                nds_pattern = r'pts_bbox_NuScenes/NDS: ([\d\.]+)'
                
                map_matches = re.findall(map_pattern, log_content)
                nds_matches = re.findall(nds_pattern, log_content)
                
                print(f"  Found {len(map_matches)} mAP values and {len(nds_matches)} NDS values")
                
                if map_matches and nds_matches:
                    # Get the latest values (assuming they're the best)
                    best_map = float(map_matches[-1])
                    best_nds = float(nds_matches[-1])
                    
                    # For the original method, use a static point at epoch 24 (completed)
                    val_data['epoch'] = [24]
                    val_data['mAP'] = [best_map]
                    val_data['NDS'] = [best_nds]
                    
                    print(f"  Using best metrics: mAP={best_map}, NDS={best_nds}")
            except Exception as ex:
                print(f"  Error in special handling: {ex}")
        
        # Extract epoch and loss information
        epoch_loss_pattern = r'mode: train.*?epoch: (\d+).*?loss: ([\d\.]+)'
        epoch_loss_matches = re.findall(epoch_loss_pattern, log_content)
        
        print(f"  Found {len(epoch_loss_matches)} training loss entries")
        
        for epoch_str, loss_str in epoch_loss_matches:
            try:
                epoch = int(epoch_str)
                loss = float(loss_str)
                if epoch not in train_data['epoch']:
                    train_data['epoch'].append(epoch)
                    train_data['train_loss'].append(loss)
            except (ValueError, TypeError):
                continue
        
        # Extract validation metrics if we haven't already filled them in special handling
        if 'epoch' not in val_data:
            val_metrics_pattern = r'mode: val.*?epoch: (\d+).*?pts_bbox_NuScenes/mAP: ([\d\.]+).*?pts_bbox_NuScenes/NDS: ([\d\.]+)'
            val_metrics_matches = re.findall(val_metrics_pattern, log_content, re.DOTALL)
            
            print(f"  Found {len(val_metrics_matches)} validation metrics entries")
            
            # If no matches found, try a more flexible pattern
            if not val_metrics_matches:
                # Try alternative patterns
                map_pattern = r'pts_bbox_NuScenes/mAP: ([\d\.]+)'
                nds_pattern = r'pts_bbox_NuScenes/NDS: ([\d\.]+)'
                epoch_pattern = r'Epoch\[(\d+)\]'
                
                map_matches = re.findall(map_pattern, log_content)
                nds_matches = re.findall(nds_pattern, log_content)
                epoch_matches = re.findall(epoch_pattern, log_content)
                
                print(f"  Alternative search found: {len(map_matches)} mAP, {len(nds_matches)} NDS, {len(epoch_matches)} epochs")
                
                # If we have valid matches and they're all the same length, combine them
                if map_matches and nds_matches and epoch_matches and len(map_matches) == len(nds_matches) and len(map_matches) <= len(epoch_matches):
                    for i in range(len(map_matches)):
                        try:
                            # Use the corresponding epoch if available, or just use the metric index
                            epoch = int(epoch_matches[i]) if i < len(epoch_matches) else i + 1
                            map_value = float(map_matches[i])
                            nds_value = float(nds_matches[i])
                            val_data['epoch'].append(epoch)
                            val_data['mAP'].append(map_value)
                            val_data['NDS'].append(nds_value)
                        except (ValueError, TypeError):
                            continue
            else:
                for epoch_str, map_str, nds_str in val_metrics_matches:
                    try:
                        epoch = int(epoch_str)
                        map_value = float(map_str)
                        nds_value = float(nds_str)
                        val_data['epoch'].append(epoch)
                        val_data['mAP'].append(map_value)
                        val_data['NDS'].append(nds_value)
                    except (ValueError, TypeError):
                        continue
        
        # Create DataFrames
        train_df = pd.DataFrame(train_data)
        val_df = pd.DataFrame(val_data)
        
        # Group val metrics by epoch and take the mean if needed
        if not val_df.empty:
            if len(val_df) > 1:  # Only group if we have multiple entries
                val_df = val_df.groupby('epoch', as_index=False).mean()
            print(f"  Final validation DataFrame has {len(val_df)} entries")
            if not val_df.empty:
                print(f"  Sample validation data: {val_df.iloc[0].to_dict()}")
        
        return train_df, val_df
        
    except Exception as e:
        print(f"Error parsing metrics from text log {log_file}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), pd.DataFrame()

# --- Main Script ---
log_files_to_process = {
    # Use shorter, clearer names for labels
    "4-Frame Adaptive": "/scratch1/ayushgoy/work_dir/4frame_adaptive/20250428_151850.log",
    "3-Frame Adaptive": "/project2/ywang234_1595/petr_v2/lbhatnag/slurm_train/petrv2_3frame_adaptive/20250428_141942.log",
    # The original method is now labeled correctly for inclusion in graphs
    "Original Method": "/project2/ywang234_1595/petr_v2/lbhatnag/slurm_train/20250418_204113.log"
}

results = {}

# Process files
for label, fpath in log_files_to_process.items():
    # Attempt to strip "SSH FS - " prefix and use the rest as the path
    # This assumes the path is either absolute or relative to the execution context
    if fpath.startswith("SSH FS - "):
        actual_path = fpath.split("SSH FS - ", 1)[1]
    else:
        actual_path = fpath

    print(f"Processing {label} ({actual_path})...")
    
    # First, try to parse config from the text log file
    config = parse_config_from_text_log(actual_path)
    
    # Then, parse metrics from the text log
    train_df, val_df = parse_metrics_from_text_log(actual_path)
    
    # If no metrics found in text log, try the JSON log as fallback
    if train_df.empty and val_df.empty:
        print(f"  No metrics found in text log, trying JSON log as fallback...")
        json_path = actual_path + ".json"
        try:
            json_train_df, json_val_df = parse_log(json_path)
            if not json_train_df.empty or not json_val_df.empty:
                print(f"  Successfully parsed metrics from JSON log")
                train_df = json_train_df
                val_df = json_val_df
        except Exception as e:
            print(f"  Error parsing JSON log: {e}")

    # Store results even if some parts are missing, checks later will handle it
    results[label] = {'train_loss': train_df, 'val_metrics': val_df, 'config': config}

    if train_df.empty and val_df.empty and "Config" not in label:
        print(f"Warning: No training or validation data found in {label} ({actual_path}). It will be excluded from plots.")
    elif train_df.empty and "Config" not in label:
        print(f"Warning: No training loss data found in {label} ({actual_path}). It will be excluded from the loss plot.")
    elif val_df.empty and "Config" not in label:
        print(f"Warning: No validation metrics found in {label} ({actual_path}). It will be excluded from the metrics plot.")
    elif "Config" in label and (train_df.empty and val_df.empty):
         print(f"Info: '{label}' only contains configuration data.")


# Filter out results that couldn't be processed or lack necessary data for plots
plot_results_metrics = {k: v for k, v in results.items() if v.get('val_metrics') is not None and not v['val_metrics'].empty}
plot_results_loss = {k: v for k, v in results.items() if v.get('train_loss') is not None and not v['train_loss'].empty}
table_results = {k: v for k, v in results.items() if v.get('config') is not None} # Include all with config for table

print("\nSummary of collected data:")
print(f"  Models with validation metrics: {list(plot_results_metrics.keys())}")
print(f"  Models with training loss: {list(plot_results_loss.keys())}")
print(f"  Models with configuration data: {list(table_results.keys())}")

# Generate plots and table if data is available
if plot_results_metrics:
    plot_metrics(plot_results_metrics)
else:
    print("\nNo validation metrics data found across files to plot.")

if plot_results_loss:
    plot_loss(plot_results_loss)
else:
    print("\nNo training loss data found across files to plot.")

if table_results:
     create_weights_table(table_results)
else:
     print("\nNo configuration data found to create weights table.")

print("\nScript finished.")