#!/usr/bin/env python
import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set the paths to the log files
log_file = "/scratch1/ayushgoy/work_dir/4frame_adaptive/20250428_151850.log"
log_json = "/scratch1/ayushgoy/work_dir/4frame_adaptive/20250428_151850.log.json"
output_dir = "/scratch1/ayushgoy/work_dir/4frame_adaptive/visualizations"

# Create the output directory
os.makedirs(output_dir, exist_ok=True)

# Function to plot training metrics from the JSON log
def plot_metrics_from_json(log_json_path, output_dir):
    print(f"Extracting metrics from {log_json_path}...")
    
    # Load the JSON log file
    try:
        with open(log_json_path, 'r') as f:
            log_data = [json.loads(line) for line in f if line.strip()]
        print(f"Successfully loaded log with {len(log_data)} entries")
    except Exception as e:
        print(f"Error loading JSON log: {e}")
        return False
    
    # Extract training metrics
    epochs = []
    loss_cls = []
    loss_bbox = []
    
    # Extract validation metrics
    val_epochs = []
    map_values = []
    nds_values = []
    
    for entry in log_data:
        # Check for training logs
        if 'mode' in entry and entry['mode'] == 'train':
            if 'epoch' in entry and 'loss_cls' in entry and 'loss_bbox' in entry:
                epochs.append(entry['epoch'])
                loss_cls.append(entry['loss_cls'])
                loss_bbox.append(entry['loss_bbox'])
        
        # Check for validation logs
        if 'mode' in entry and entry['mode'] == 'val':
            if 'epoch' in entry and 'pts_bbox_NuScenes/mAP' in entry and 'pts_bbox_NuScenes/NDS' in entry:
                val_epochs.append(entry['epoch'])
                map_values.append(entry['pts_bbox_NuScenes/mAP'])
                nds_values.append(entry['pts_bbox_NuScenes/NDS'])
    
    if not epochs:
        print("No training metrics found in the log")
        return False
    
    # Plot training losses
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, loss_cls, 'b-', label='Classification Loss')
    plt.plot(epochs, loss_bbox, 'r-', label='Bounding Box Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses for 4-Frame Adaptive PETR')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_losses.png'), dpi=300)
    plt.close()
    
    # Plot validation metrics if available
    if val_epochs:
        plt.figure(figsize=(12, 6))
        plt.plot(val_epochs, map_values, 'g-', label='mAP')
        plt.plot(val_epochs, nds_values, 'm-', label='NDS')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Validation Metrics for 4-Frame Adaptive PETR')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'validation_metrics.png'), dpi=300)
        plt.close()
        
        # Print the best metrics
        best_epoch_idx = np.argmax(nds_values)
        best_epoch = val_epochs[best_epoch_idx]
        best_map = map_values[best_epoch_idx]
        best_nds = nds_values[best_epoch_idx]
        
        print(f"Best validation results at epoch {best_epoch}:")
        print(f"  mAP: {best_map:.4f}")
        print(f"  NDS: {best_nds:.4f}")
    
    return True

# Function to extract metrics from the text log using regex
def plot_metrics_from_text_log(log_file_path, output_dir):
    print(f"Extracting metrics from text log {log_file_path}...")
    
    try:
        with open(log_file_path, 'r') as f:
            log_content = f.read()
        print("Successfully loaded text log")
    except Exception as e:
        print(f"Error loading text log: {e}")
        return False
    
    # Extract map and nds from evaluation results
    map_pattern = r"pts_bbox_NuScenes/mAP: (\d+\.\d+)"
    nds_pattern = r"pts_bbox_NuScenes/NDS: (\d+\.\d+)"
    
    map_matches = re.findall(map_pattern, log_content)
    nds_matches = re.findall(nds_pattern, log_content)
    
    if not map_matches or not nds_matches:
        print("No evaluation metrics found in text log")
        return False
    
    # Extract per-class AP (focusing on the last evaluation)
    class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 
                  'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    
    class_ap_pattern = r"pts_bbox_NuScenes/{}_AP_dist_2.0: (\d+\.\d+)"
    
    class_ap_values = {}
    for class_name in class_names:
        pattern = class_ap_pattern.format(class_name)
        matches = re.findall(pattern, log_content)
        if matches:
            # Take the last value (most recent evaluation)
            class_ap_values[class_name] = float(matches[-1])
    
    # Create class AP bar chart
    if class_ap_values:
        plt.figure(figsize=(12, 6))
        classes = list(class_ap_values.keys())
        ap_values = [class_ap_values[c] for c in classes]
        
        # Sort by AP value
        sorted_indices = np.argsort(ap_values)
        sorted_classes = [classes[i] for i in sorted_indices]
        sorted_ap_values = [ap_values[i] for i in sorted_indices]
        
        # Create bar chart with color gradient
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0.2, 0.8, len(sorted_classes)))
        
        bars = plt.barh(sorted_classes, sorted_ap_values, color=colors)
        plt.xlabel('AP @ 2.0m')
        plt.title('Per-Class Performance (AP @ 2.0m distance)')
        plt.grid(axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_class_performance.png'), dpi=300)
        plt.close()
        
        # Print class AP values
        print("Per-class AP @ 2.0m:")
        for class_name, ap in sorted(class_ap_values.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {ap:.4f}")
    
    return True

# Function to visualize the adaptive weights concept
def visualize_adaptive_weights_concept(output_dir):
    print("Creating adaptive weights concept visualization...")
    
    # Create a directory for the adaptive weights visualizations
    adaptive_dir = os.path.join(output_dir, 'adaptive_weights')
    os.makedirs(adaptive_dir, exist_ok=True)
    
    # Set up frames
    frames = ['Current Frame', 'Previous Frame 1', 'Previous Frame 2', 'Previous Frame 3']
    
    # Different scenarios of adaptive weights
    scenarios = [
        {
            'title': 'Fast Motion Scene - Higher Weight on Current Frame',
            'weights': [0.55, 0.25, 0.15, 0.05],
            'description': 'In scenes with fast motion, the model assigns more weight to the current frame as previous frames contain less relevant information.'
        },
        {
            'title': 'Moderate Motion - Balanced Weights',
            'weights': [0.40, 0.30, 0.20, 0.10],
            'description': 'With moderate motion, the model balances between current and recent previous frames.'
        },
        {
            'title': 'Static Scene - Even Distribution',
            'weights': [0.30, 0.28, 0.25, 0.17],
            'description': 'In static scenes, the model can utilize information from all frames more evenly.'
        },
        {
            'title': 'Occlusion Scenario - Higher Weight on Previous Frames',
            'weights': [0.25, 0.35, 0.25, 0.15],
            'description': 'When objects are occluded in the current frame, previous frames may provide better information.'
        }
    ]
    
    # Create plots for each scenario
    for i, scenario in enumerate(scenarios):
        plt.figure(figsize=(10, 6))
        
        # Create bar plot with custom colors
        sns.set_style("whitegrid")
        bars = plt.bar(frames, scenario['weights'], 
                      color=['#ff5555', '#ff8855', '#ffbb55', '#ffee55'])
        
        # Add value labels on top of bars
        for bar, weight in zip(bars, scenario['weights']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{weight:.2f}', ha='center', fontweight='bold')
        
        plt.ylim(0, 0.6)
        plt.title(scenario['title'], fontsize=15, fontweight='bold')
        plt.xlabel('Frames', fontsize=12)
        plt.ylabel('Adaptive Weight', fontsize=12)
        
        # Add description as text box
        plt.figtext(0.5, 0.01, scenario['description'], ha='center', 
                   fontsize=10, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        plt.savefig(os.path.join(adaptive_dir, f'adaptive_weights_scenario_{i+1}.png'), dpi=300)
        plt.close()
    
    # Create a visualization of adaptive fusion architecture
    plt.figure(figsize=(12, 8))
    
    # Define the architecture components
    components = [
        {'name': 'Current Frame', 'x': 0.2, 'y': 0.8, 'width': 0.15, 'height': 0.1, 'color': '#ff5555'},
        {'name': 'Previous Frame 1', 'x': 0.2, 'y': 0.65, 'width': 0.15, 'height': 0.1, 'color': '#ff8855'},
        {'name': 'Previous Frame 2', 'x': 0.2, 'y': 0.5, 'width': 0.15, 'height': 0.1, 'color': '#ffbb55'},
        {'name': 'Previous Frame 3', 'x': 0.2, 'y': 0.35, 'width': 0.15, 'height': 0.1, 'color': '#ffee55'},
        
        {'name': 'Feature Extraction', 'x': 0.4, 'y': 0.575, 'width': 0.2, 'height': 0.25, 'color': '#aaddff'},
        
        {'name': 'Adaptive Weight\nCalculation', 'x': 0.7, 'y': 0.7, 'width': 0.2, 'height': 0.15, 'color': '#ffccaa'},
        
        {'name': 'Weighted\nFeature Fusion', 'x': 0.7, 'y': 0.45, 'width': 0.2, 'height': 0.15, 'color': '#aaffcc'},
        
        {'name': '3D Object\nDetection', 'x': 0.7, 'y': 0.2, 'width': 0.2, 'height': 0.15, 'color': '#ddaaff'}
    ]
    
    # Add the components as rectangles
    ax = plt.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    for comp in components:
        rect = plt.Rectangle((comp['x'], comp['y']), comp['width'], comp['height'],
                           facecolor=comp['color'], edgecolor='black', alpha=0.8,
                           linewidth=2, zorder=1)
        ax.add_patch(rect)
        plt.text(comp['x'] + comp['width']/2, comp['y'] + comp['height']/2,
                comp['name'], ha='center', va='center', fontweight='bold',
                fontsize=10, zorder=2)
    
    # Add arrows to connect components
    arrows = [
        # From frames to feature extraction
        {'start': (0.35, 0.85), 'end': (0.4, 0.7), 'color': '#ff5555'},
        {'start': (0.35, 0.7), 'end': (0.4, 0.65), 'color': '#ff8855'},
        {'start': (0.35, 0.55), 'end': (0.4, 0.6), 'color': '#ffbb55'},
        {'start': (0.35, 0.4), 'end': (0.4, 0.55), 'color': '#ffee55'},
        
        # From feature extraction to adaptive weight calculation
        {'start': (0.6, 0.7), 'end': (0.7, 0.75), 'color': 'black'},
        
        # From feature extraction to weighted fusion
        {'start': (0.6, 0.6), 'end': (0.7, 0.5), 'color': 'black'},
        
        # From adaptive weights to weighted fusion
        {'start': (0.8, 0.7), 'end': (0.8, 0.6), 'color': 'black'},
        
        # From weighted fusion to detection
        {'start': (0.8, 0.45), 'end': (0.8, 0.35), 'color': 'black'}
    ]
    
    for arrow in arrows:
        ax.annotate('', xy=arrow['end'], xytext=arrow['start'],
                   arrowprops=dict(facecolor=arrow['color'], shrink=0.05, width=1.5,
                                  headwidth=7, alpha=0.8), zorder=3)
    
    plt.title('4-Frame Adaptive PETRv2 Architecture', fontsize=16, fontweight='bold', pad=20)
    
    # Add explanation
    explanation = """
    The 4-Frame Adaptive PETRv2 processes the current frame along with 3 previous frames.
    Features are extracted from all frames and passed to an adaptive weighting module,
    which determines the optimal contribution of each frame based on motion patterns.
    The weighted features are fused and used for 3D object detection.
    """
    
    plt.figtext(0.5, 0.05, explanation, ha='center', va='center',
               fontsize=11, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig(os.path.join(adaptive_dir, 'adaptive_fusion_architecture.png'), dpi=300)
    plt.close()
    
    print(f"Adaptive weights visualizations saved to {adaptive_dir}")
    return True

# Main execution
print(f"Starting visualization for 4-Frame Adaptive PETR...")
print(f"Log file: {log_file}")
print(f"Log JSON: {log_json}")
print(f"Output directory: {output_dir}")

# Plot metrics from JSON log
json_success = plot_metrics_from_json(log_json, output_dir)

# If JSON parsing fails, try text log
if not json_success:
    plot_metrics_from_text_log(log_file, output_dir)

# Create adaptive weights visualization
visualize_adaptive_weights_concept(output_dir)

print(f"Visualization complete! Results saved to: {output_dir}") 