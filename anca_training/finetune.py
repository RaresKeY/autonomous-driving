# finetune.py
# Fine-tuning YOLOv8 pe KITTI pentru Car, Pedestrian, Cyclist
# Anca - Model Training Engineer

import os
from pathlib import Path
from ultralytics import YOLO
import yaml


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def finetune(dataset_yaml, run_name="kitti_finetune_v1"):
    """
    Fine-tuneaza YOLOv8 pe datasetul KITTI convertit.

    Args:
        dataset_yaml: calea catre dataset.yaml generat de prepare_kitti.py
        run_name: numele experimentului (se salveaza in runs/)
    """
    cfg = load_config()

    # Verifica ca dataset.yaml exista
    if not Path(dataset_yaml).exists():
        print(f"EROARE: Nu gasesc {dataset_yaml}")
        print("Ruleaza mai intai prepare_kitti.py!")
        return

    print("=" * 40)
    print("INCEPE FINE-TUNING")
    print("=" * 40)
    print(f"Dataset:  {dataset_yaml}")
    print(f"Model:    yolov8n.pt (pre-antrenat COCO)")
    print(f"Epoci:    {cfg['training']['epochs']}")
    print(f"Batch:    {cfg['training']['batch_size']}")
    print(f"Device:   {cfg['training']['device']}")
    print()

    # Incarca modelul pre-antrenat
    # Pornim de la yolov8n.pt care stie deja forme, margini, texturi
    # Fine-tuning = il invatam sa recunoasca specific Car/Pedestrian/Cyclist
    model = YOLO("yolov8n.pt")

    # Antrenare
    results = model.train(
        data=dataset_yaml,
        epochs=cfg["training"]["epochs"],
        batch=cfg["training"]["batch_size"],
        imgsz=cfg["data"]["img_size"],
        lr0=cfg["training"]["learning_rate"],
        lrf=0.01,           # learning rate final (scade treptat)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,    # primele 3 epoci cu lr mic, sa nu distruga weights
        device=cfg["training"]["device"],
        project="runs",
        name=run_name,
        exist_ok=True,

        # Augmentari - ajuta modelul sa generalizeze
        hsv_h=0.015,        # variatie culoare
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,         # nu intoarcem imaginile cu susul in jos (nu are sens la masini)
        fliplr=0.5,         # intoarcem stanga-dreapta (are sens)
        mosaic=1.0,         # combina 4 imagini intr-una (augmentare puternica)
        mixup=0.1,          # amesteca 2 imagini

        # Salvare
        save=True,
        save_period=5,      # salveaza checkpoint la fiecare 5 epoci
        val=True,           # evalueaza pe val dupa fiecare epoca
    )

    # Afiseaza rezultatele finale
    print("\n" + "=" * 40)
    print("FINE-TUNING FINALIZAT")
    print("=" * 40)

    best_model_path = f"runs/{run_name}/weights/best.pt"
    last_model_path = f"runs/{run_name}/weights/last.pt"

    if Path(best_model_path).exists():
        print(f"Best model: {best_model_path}")
        print(f"Last model: {last_model_path}")
        print()
        print("Metrici finale:")
        print(f"  mAP@50:    {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.3f}")
        print(f"  mAP@50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.3f}")
        print(f"  Precision: {results.results_dict.get('metrics/precision(B)', 'N/A'):.3f}")
        print(f"  Recall:    {results.results_dict.get('metrics/recall(B)', 'N/A'):.3f}")
    else:
        print("Modelul nu a fost salvat corect, verifica erorile de mai sus.")

    return results


def evaluate(dataset_yaml, model_path="runs/kitti_finetune_v1/weights/best.pt"):
    """
    Evalueaza modelul antrenat pe setul de test.
    Ruleaza dupa fine-tuning pentru raportul final.
    """
    if not Path(model_path).exists():
        print(f"EROARE: Nu gasesc modelul la {model_path}")
        return

    print(f"Evaluez modelul: {model_path}")
    model = YOLO(model_path)

    results = model.val(
        data=dataset_yaml,
        split="test",       # folosim setul de test, nu val
        imgsz=640,
        device="cpu",
    )

    print("\nREZULTATE PE TEST SET:")
    print(f"  mAP@50:    {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.3f}")
    print(f"  mAP@50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.3f}")
    print(f"  Precision: {results.results_dict.get('metrics/precision(B)', 'N/A'):.3f}")
    print(f"  Recall:    {results.results_dict.get('metrics/recall(B)', 'N/A'):.3f}")

    return results


if __name__ == "__main__":
    DATASET_YAML = r"C:\Curs Python\datasets\kitti_yolo\dataset.yaml"

    # Pasul 1: Fine-tuning
    finetune(DATASET_YAML, run_name="kitti_finetune_v1")

    # Pasul 2: Evaluare pe test set
    evaluate(DATASET_YAML, model_path="runs/kitti_finetune_v1/weights/best.pt")