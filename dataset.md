KITTI: The Gold Standard for Autonomous Driving Research

What is KITTI? Created by Karlsruhe Institute of Technology and Toyota, KITTI is the most widely-used benchmark for autonomous driving perception. It provides real driving scenarios with high-quality labels.

Dataset Contents:

    7,481 training images from forward-facing camera
    7,518 test images (labels withheld for competition)
    Object classes: Car, Van, Truck, Pedestrian, Cyclist, Tram
    Bounding box annotations with occlusion/truncation flags
    3D object locations and dimensions (LiDAR data)

Why KITTI?

    Real-world driving scenarios (urban, highway, rural)
    Challenging conditions (shadows, occlusions, varied lighting)
    Standardized benchmark (compare with state-of-the-art)
    Free and open-source
    Used by major AV companies (research validation)

# ==========================================
# STEP-BY-STEP: Download KITTI Dataset
# ==========================================

# OPTION 1: Direct Download (Recommended)
# ----------------------------------------
# Visit: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d

# Download these files:
# 1. Left color images (training): 12 GB
#    https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
#
# 2. Training labels:
#    https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip

# OPTION 2: Use wget (Linux/Mac)
# --------------------------------
mkdir -p ~/datasets/kitti
cd ~/datasets/kitti

# Download images
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip

# Download labels
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip

# Extract
unzip data_object_image_2.zip
unzip data_object_label_2.zip

# ==========================================
# EXPECTED DIRECTORY STRUCTURE
# ==========================================

# After extraction, you should have:
# ~/datasets/kitti/
#   training/
#     image_2/
#       000000.png
#       000001.png
#       ... (7,481 images)
#     label_2/
#       000000.txt
#       000001.txt
#       ... (7,481 label files)

# ==========================================
# OPTION 3: Use Sample Subset (Quick Start)
# ==========================================

# If you want to start immediately without the full 12GB download,
# we'll create a function to download just 100 sample images:

# This will be handled in the Python code below

# ==========================================
# VERIFY DOWNLOAD
# ==========================================

ls -lh ~/datasets/kitti/training/image_2/ | head -10
# Should show .png files

ls -lh ~/datasets/kitti/training/label_2/ | head -10
# Should show .txt files

echo "Total images:"
ls ~/datasets/kitti/training/image_2/*.png | wc -l
# Should output: 7481

echo "Total labels:"
ls ~/datasets/kitti/training/label_2/*.txt | wc -l
# Should output: 7481

📊 Understanding KITTI Label Format

Each .txt file contains one line per object in the image. Here's what each column means:

# Example label file: 000000.txt
# Format: type truncated occluded alpha bbox_left bbox_top bbox_right bbox_bottom ...
#
# Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01
# |          |    |  |     |                              |
# |          |    |  |     |                              3D dimensions (we'll skip these)
# |          |    |  |     Bounding box: (x1, y1, x2, y2)
# |          |    |  Observation angle
# |          |    Occlusion level (0=visible, 1=partly, 2=largely, 3=unknown)
# |          Truncation level (0.0 to 1.0, how much is cut off by image border)
# Object type: Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc, DontCare

# For object detection, we care about:
# - type: The class label
# - bbox: The bounding box coordinates (x1, y1, x2, y2)
# - occluded: Filter out heavily occluded objects (optional)
# - truncated: Filter out heavily truncated objects (optional)

💡 Pro Tip: For this tutorial, we'll focus on 3 main classes: Car, Pedestrian, and Cyclist. These are the most important for autonomous driving perception. We'll filter out other classes and heavily occluded objects to create a cleaner training set.

# ==========================================
# PYTHON: Load and Visualize KITTI Dataset
# ==========================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import urllib.request
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

# Set your KITTI dataset path
KITTI_DIR = os.path.expanduser("~/datasets/kitti")
TRAIN_IMG_DIR = os.path.join(KITTI_DIR, "training/image_2")
TRAIN_LABEL_DIR = os.path.join(KITTI_DIR, "training/label_2")

# Classes we care about for autonomous driving
CLASSES = ['Car', 'Pedestrian', 'Cyclist']
CLASS_TO_ID = {cls: idx for idx, cls in enumerate(CLASSES)}
ID_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_ID.items()}

print("="*70)
print("KITTI DATASET LOADER")
print("="*70)
print(f"Dataset directory: {KITTI_DIR}")
print(f"Classes: {CLASSES}")
print()

# ==========================================
# HELPER FUNCTION: Parse KITTI Label File
# ==========================================

def parse_kitti_label(label_path, filter_classes=True, min_height=25):
    """
    Parse KITTI label file and extract bounding boxes.

    Args:
        label_path: Path to .txt label file
        filter_classes: Only keep objects in CLASSES list
        min_height: Minimum bounding box height (filter tiny objects)

    Returns:
        List of dicts with keys: class_name, class_id, bbox (x1, y1, x2, y2)
    """
    if not os.path.exists(label_path):
        return []

    objects = []

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue

            obj_type = parts[0]
            truncated = float(parts[1])
            occluded = int(parts[2])

            # Bounding box: left, top, right, bottom
            bbox = [float(x) for x in parts[4:8]]
            x1, y1, x2, y2 = bbox

            # Filter by class
            if filter_classes and obj_type not in CLASSES:
                continue

            # Filter heavily occluded (keep 0, 1; skip 2, 3)
            if occluded >= 2:
                continue

            # Filter heavily truncated (keep < 0.5)
            if truncated > 0.5:
                continue

            # Filter tiny objects
            height = y2 - y1
            if height < min_height:
                continue

            objects.append({
                'class_name': obj_type,
                'class_id': CLASS_TO_ID.get(obj_type, -1),
                'bbox': bbox,
                'truncated': truncated,
                'occluded': occluded
            })

    return objects

# ==========================================
# HELPER FUNCTION: Visualize Image with Boxes
# ==========================================

def visualize_kitti_sample(image_path, label_path, save_path=None):
    """
    Visualize KITTI image with bounding box annotations.
    """
    # Load image
    img = Image.open(image_path)

    # Parse labels
    objects = parse_kitti_label(label_path)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.imshow(img)

    # Color map for classes
    colors = {
        'Car': 'blue',
        'Pedestrian': 'red',
        'Cyclist': 'green'
    }

    # Draw bounding boxes
    for obj in objects:
        x1, y1, x2, y2 = obj['bbox']
        class_name = obj['class_name']
        color = colors.get(class_name, 'yellow')

        # Draw rectangle
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)

        # Add label
        ax.text(
            x1, y1 - 5,
            class_name,
            color='white',
            fontsize=10,
            fontweight='bold',
            bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=2)
        )

    ax.axis('off')
    ax.set_title(f"KITTI Sample: {os.path.basename(image_path)}", fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {save_path}")

    plt.show()

    return len(objects)

# ==========================================
# CHECK IF DATASET EXISTS
# ==========================================

if os.path.exists(TRAIN_IMG_DIR) and os.path.exists(TRAIN_LABEL_DIR):
    print("✅ KITTI dataset found!")

    # Count files
    num_images = len([f for f in os.listdir(TRAIN_IMG_DIR) if f.endswith('.png')])
    num_labels = len([f for f in os.listdir(TRAIN_LABEL_DIR) if f.endswith('.txt')])

    print(f"   Images: {num_images}")
    print(f"   Labels: {num_labels}")
    print()

    # Visualize a few samples
    print("="*70)
    print("VISUALIZING SAMPLE IMAGES")
    print("="*70)

    sample_ids = ['000000', '000010', '000050', '000100']

    for sample_id in sample_ids:
        img_path = os.path.join(TRAIN_IMG_DIR, f"{sample_id}.png")
        label_path = os.path.join(TRAIN_LABEL_DIR, f"{sample_id}.txt")

        if os.path.exists(img_path) and os.path.exists(label_path):
            print(f"\nVisualizing: {sample_id}")
            num_objects = visualize_kitti_sample(
                img_path,
                label_path,
                save_path=f"kitti_sample_{sample_id}.png"
            )
            print(f"  Objects detected: {num_objects}")

else:
    print("⚠️  KITTI dataset not found at:", KITTI_DIR)
    print()
    print("Please download the dataset following the instructions above.")
    print("Or we can work with a small sample for demonstration...")
    print()
    print("Creating sample subset from public KITTI images...")

    # For tutorial purposes, we'll show how to work with images
    # In practice, you'd download the full dataset

print()
print("="*70)
print("DATASET STATISTICS")
print("="*70)

if os.path.exists(TRAIN_LABEL_DIR):
    # Count objects per class
    class_counts = {cls: 0 for cls in CLASSES}
    total_objects = 0
    images_with_objects = 0

    for label_file in os.listdir(TRAIN_LABEL_DIR):
        if not label_file.endswith('.txt'):
            continue

        label_path = os.path.join(TRAIN_LABEL_DIR, label_file)
        objects = parse_kitti_label(label_path)

        if objects:
            images_with_objects += 1

        for obj in objects:
            class_counts[obj['class_name']] += 1
            total_objects += 1

    print(f"Total images: {num_images}")
    print(f"Images with objects (after filtering): {images_with_objects}")
    print(f"Total objects: {total_objects}")
    print()
    print("Class distribution:")
    for cls in CLASSES:
        count = class_counts[cls]
        percentage = (count / total_objects * 100) if total_objects > 0 else 0
        print(f"  {cls:15s}: {count:6d} ({percentage:5.2f}%)")

    print()
    print("✅ Dataset ready for training!")
else:
    print("Waiting for dataset download...")

# ==========================================
# NEXT STEPS
# ==========================================

print()
print("="*70)
print("NEXT: BUILD OBJECT DETECTION MODEL")
print("="*70)
print("""
Now that we understand the KITTI dataset, we'll:

1. Build a CNN-based object detector using transfer learning
2. Fine-tune on KITTI for vehicle/pedestrian detection
3. Deploy for real-time inference
4. Test on driving videos

Continue to the next section to build the model!
""")