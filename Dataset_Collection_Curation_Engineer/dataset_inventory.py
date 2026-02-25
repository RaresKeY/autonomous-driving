import os
import random
import json
from collections import Counter

print("\n----- DATASET CHECK START -----\n")

IMAGE_DIR = r"C:\Users\Claudia Tatucu\datasets\kitti\training\image_2"
LABEL_DIR = r"C:\Users\Claudia Tatucu\datasets\kitti\training\label_2"

# 1. Găsește imaginile (.png la KITTI)
images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".png")]

# 2. Găsește label-urile
labels = [f for f in os.listdir(LABEL_DIR) if f.endswith(".txt")]

# Verificare dataset gol
if len(images) == 0:
    print("No images found. Exiting.")
    exit(1)

# 3. Verificare integritate
image_names = {os.path.splitext(f)[0] for f in images}
label_names = {os.path.splitext(f)[0] for f in labels}

missing_labels = image_names - label_names
missing_images = label_names - image_names

# 4. Numărare clase (prima coloană din fiecare linie KITTI)
classes = []

for file in labels:
    with open(os.path.join(LABEL_DIR, file), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 0:
                classes.append(parts[0])

class_counts = Counter(classes)

# 5. Split dataset
random.shuffle(images)

train_end = int(0.7 * len(images))
val_end = int(0.85 * len(images))

train = images[:train_end]
val = images[train_end:val_end]
test = images[val_end:]

# 6. Salvare split
with open("train.txt", "w", encoding="utf-8") as f:
    for item in train:
        f.write(item + "\n")

with open("val.txt", "w", encoding="utf-8") as f:
    for item in val:
        f.write(item + "\n")

with open("test.txt", "w", encoding="utf-8") as f:
    for item in test:
        f.write(item + "\n")

# 7. Raport final în consolă
print("----- DATA INVENTORY REPORT -----")
print(f"Total images: {len(images)}")
print(f"Total labels: {len(labels)}")
print(f"Class distribution: {class_counts}")
print(f"Missing labels: {missing_labels}")
print(f"Missing images: {missing_images}")
print("---------------------------------\n")

# 8. Salvare raport text
with open("data_inventory_report.txt", "w", encoding="utf-8") as f:
    f.write("----- DATA INVENTORY REPORT -----\n")
    f.write(f"Total images: {len(images)}\n")
    f.write(f"Total labels: {len(labels)}\n")
    f.write(f"Class distribution: {class_counts}\n")
    f.write(f"Missing labels: {missing_labels}\n")
    f.write(f"Missing images: {missing_images}\n")
    f.write("---------------------------------\n")

# 9. Manifest JSON
manifest = {
    "dataset_name": "KITTI",
    "license": "KITTI Vision Benchmark Suite - see official website for usage terms",
    "image_directory": IMAGE_DIR,
    "label_directory": LABEL_DIR,
    "total_images": len(images),
    "total_labels": len(labels),
    "class_distribution": dict(class_counts),
    "missing_labels": list(missing_labels),
    "missing_images": list(missing_images),
    "split": {
        "train": len(train),
        "val": len(val),
        "test": len(test)
    }
}

with open("dataset_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=4)

print("Generated files:")
print("- train.txt")
print("- val.txt")
print("- test.txt")
print("- data_inventory_report.txt")
print("- dataset_manifest.json\n")