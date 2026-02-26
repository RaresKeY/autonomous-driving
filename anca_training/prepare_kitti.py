# prepare_kitti.py
# Converteste datele KITTI in formatul YOLO pentru fine-tuning
# Anca - Model Training Engineer

import os
import shutil
import random
from pathlib import Path

# Clasele noastre - KITTI are mai multe clase, noi vrem doar astea 3
KITTI_TO_YOLO = {
    "Car":        0,
    "Pedestrian": 1,
    "Cyclist":    2,
    # Ignoram: Van, Truck, Person_sitting, Tram, Misc, DontCare
}

# Splituri train/val/test
SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}


def parse_kitti_label(label_path):
    """
    Citeste un fisier .txt KITTI si returneaza lista de obiecte.
    Format KITTI: class truncated occluded alpha x1 y1 x2 y2 ...
    """
    objects = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            cls_name = parts[0]
            if cls_name not in KITTI_TO_YOLO:
                continue  # ignoram clasele care nu ne intereseaza

            # Coordonate bounding box in pixeli (format KITTI)
            x1, y1, x2, y2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

            objects.append({
                "class_id": KITTI_TO_YOLO[cls_name],
                "class_name": cls_name,
                "bbox_pixels": (x1, y1, x2, y2)
            })

    return objects


def convert_bbox_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """
    Converteste din format KITTI (pixeli absoluti) in format YOLO (0-1 normalizat).
    YOLO vrea: x_center, y_center, width, height (toate intre 0 si 1)
    """
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width    = (x2 - x1) / img_w
    height   = (y2 - y1) / img_h
    return x_center, y_center, width, height


def prepare_dataset(kitti_images_dir, kitti_labels_dir, output_dir, img_w=1242, img_h=375):
    """
    Converteste intregul dataset KITTI in structura YOLO.
    
    Args:
        kitti_images_dir: folderul cu pozele KITTI (.png)
        kitti_labels_dir: folderul cu etichetele KITTI (.txt)
        output_dir: unde salvam datasetul convertit
        img_w, img_h: dimensiunile standard ale imaginilor KITTI
    """
    kitti_images_dir = Path(kitti_images_dir)
    kitti_labels_dir = Path(kitti_labels_dir)
    output_dir = Path(output_dir)

    # Gaseste toate imaginile care au si eticheta
    image_files = sorted(kitti_images_dir.glob("*.png"))
    valid_pairs = []

    for img_path in image_files:
        label_path = kitti_labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            valid_pairs.append((img_path, label_path))

    print(f"Gasit {len(valid_pairs)} perechi imagine+eticheta")

    # Shuffle + split
    random.seed(42)
    random.shuffle(valid_pairs)

    n = len(valid_pairs)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val   = int(n * SPLIT_RATIOS["val"])

    splits = {
        "train": valid_pairs[:n_train],
        "val":   valid_pairs[n_train:n_train + n_val],
        "test":  valid_pairs[n_train + n_val:]
    }

    # Statistici clase
    class_counts = {"Car": 0, "Pedestrian": 0, "Cyclist": 0}

    # Proceseaza fiecare split
    for split_name, pairs in splits.items():
        img_out = output_dir / split_name / "images"
        lbl_out = output_dir / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        print(f"\nProceseaza {split_name}: {len(pairs)} imagini...")

        for img_path, label_path in pairs:
            # Copiaza imaginea
            shutil.copy(img_path, img_out / img_path.name)

            # Converteste etichetele
            objects = parse_kitti_label(label_path)
            yolo_lines = []

            for obj in objects:
                x1, y1, x2, y2 = obj["bbox_pixels"]
                xc, yc, w, h = convert_bbox_to_yolo(x1, y1, x2, y2, img_w, img_h)
                yolo_lines.append(f"{obj['class_id']} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                class_counts[obj["class_name"]] += 1

            # Salveaza eticheta in format YOLO
            out_label = lbl_out / (img_path.stem + ".txt")
            with open(out_label, "w") as f:
                f.write("\n".join(yolo_lines))

    # Genereaza fisierul dataset.yaml (necesar pentru YOLO training)
    yaml_content = f"""# Dataset KITTI - generat automat de prepare_kitti.py
path: {output_dir.resolve()}
train: train/images
val: val/images
test: test/images

nc: 3
names:
  0: Car
  1: Pedestrian
  2: Cyclist
"""
    with open(output_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content)

    # Raport final
    print("\n" + "=" * 40)
    print("DATASET PREGATIT")
    print("=" * 40)
    print(f"Train:  {len(splits['train'])} imagini")
    print(f"Val:    {len(splits['val'])} imagini")
    print(f"Test:   {len(splits['test'])} imagini")
    print(f"\nDistributie clase:")
    for cls, count in class_counts.items():
        bar = "█" * min(count // 10, 40)
        print(f"  {cls:<12} {count:>5}x  {bar}")
    print(f"\nSalvat in: {output_dir}")
    print(f"YAML:      {output_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    # Modifica aceste path-uri cand ai datele descarcate
    KITTI_IMAGES = r"C:\Curs Python\datasets\kitti\images"
    KITTI_LABELS = r"C:\Curs Python\datasets\kitti\labels"
    OUTPUT_DIR   = r"C:\Curs Python\datasets\kitti_yolo"

    prepare_dataset(KITTI_IMAGES, KITTI_LABELS, OUTPUT_DIR)