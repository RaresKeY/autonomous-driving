# train.py
# Scriptul principal de antrenare - Anca
# Model Training Engineer

import yaml
from ultralytics import YOLO

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train():
    # 1. Incarca configurarea
    cfg = load_config()
    print("Config incarcata OK")
    print(f"  Model: {cfg['model']['architecture']}")
    print(f"  Epoci: {cfg['training']['epochs']}")
    print(f"  Device: {cfg['training']['device']}")

    # 2. Incarca modelul YOLOv8 pre-antrenat
    model_name = cfg['model']['architecture'] + ".pt"
    model = YOLO(model_name)
    print(f"\nModel {model_name} incarcat OK")

    # 3. Antreneaza pe COCO128 (dataset mic de test, built-in in ultralytics)
    # Cand Mihaela termina dataloader-ul, inlocuim 'coco128.yaml' cu datele reale
    print("\nIncep antrenarea (mock cu coco128)...\n")
    results = model.train(
        data="coco128.yaml",                    # <-- se va inlocui cu KITTI
        epochs=cfg['training']['epochs'],
        batch=cfg['training']['batch_size'],
        imgsz=cfg['data']['img_size'],
        lr0=cfg['training']['learning_rate'],
        device=cfg['training']['device'],
        name=cfg['output']['run_name'],
        project=cfg['output']['save_dir'],
    )

    print("\nAntrenare finalizata!")
    print(f"Rezultate salvate in: runs/{cfg['output']['run_name']}/")
    return results

if __name__ == "__main__":
    train()