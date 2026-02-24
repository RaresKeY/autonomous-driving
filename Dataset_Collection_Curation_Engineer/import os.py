import os
import random
from collections import Counter

print("\n----- DATASET CHECK START -----\n")

# 1. Găsește imaginile
files = os.listdir(".")
images = [f for f in files if f.endswith(".jpg")]

# 2. Găsește label-urile
labels = [f for f in files if f.endswith(".txt") and f not in ["train.txt", "val.txt", "test.txt"]]

# 3. Verificare integritate
image_names = {os.path.splitext(f)[0] for f in images}
label_names = {os.path.splitext(f)[0] for f in labels}

missing_labels = image_names - label_names
missing_images = label_names - image_names

# 4. Numărare clase
classes = []
for file in labels:
    with open(file, "r", encoding="utf-8") as f:
        classes.append(f.read().strip())

class_counts = Counter(classes)

# 5. Split dataset
random.shuffle(images)

train_end = int(0.7 * len(images))
val_end = int(0.85 * len(images))

train = images[:train_end]
val = images[train_end:val_end]
test = images[val_end:]

# 6. Salvare split
with open("train.txt", "w") as f:
    for item in train:
        f.write(item + "\n")

with open("val.txt", "w") as f:
    for item in val:
        f.write(item + "\n")

with open("test.txt", "w") as f:
    for item in test:
        f.write(item + "\n")

# 7. Raport final
print("----- DATA INVENTORY REPORT -----")
print(f"Total images: {len(images)}")
print(f"Class distribution: {class_counts}")
print(f"Missing labels: {missing_labels}")
print(f"Missing images: {missing_images}")
print("---------------------------------\n")