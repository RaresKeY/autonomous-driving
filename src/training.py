# src/training.py
# Pipeline de training reproducibil - Anca
# Model Training Engineer

import argparse
import sys
from pathlib import Path


# ============================================================
# 1. BUILD MODEL
# ============================================================

def build_model(architecture="yolov8n", pretrained=True):
    """
    Construieste si returneaza modelul YOLOv8.
    Separata de training ca sa poata fi reutilizata si de Paul la inferenta.

    Args:
        architecture: numele modelului (yolov8n, yolov8s, yolov8m)
        pretrained: daca pornim de la weights pre-antrenate

    Returns:
        model YOLO gata de antrenat sau de inferenta
    """
    try:
        from ultralytics import YOLO

        model_file = f"{architecture}.pt" if pretrained else architecture
        model = YOLO(model_file)
        print(f"[build_model] Model '{model_file}' incarcat OK")
        return model

    except Exception as e:
        print(f"[build_model] EROARE la incarcarea modelului: {e}")
        raise


# ============================================================
# 2. TRAIN MODEL (entrypoint principal)
# ============================================================

def train_model(
    dataset_path,
    epochs=10,
    batch_size=4,
    img_size=640,
    learning_rate=0.01,
    architecture="yolov8n",
    output_dir="runs",
    run_name="experiment_v1",
    device="cpu",
):
    """
    Entrypoint principal de training. Apelabil din CLI si din cod.

    Args:
        dataset_path : calea catre dataset.yaml (KITTI convertit)
        epochs       : numarul de epoci
        batch_size   : marimea batch-ului
        img_size     : dimensiunea imaginilor
        learning_rate: learning rate initial
        architecture : arhitectura YOLOv8
        output_dir   : folderul unde se salveaza runs
        run_name     : numele experimentului
        device       : 'cpu' sau '0' pentru GPU

    Returns:
        dict cu caile catre artefacte:
        {
            "best_model":  "runs/experiment_v1/weights/best.pt",
            "final_model": "runs/experiment_v1/weights/last.pt",
            "run_dir":     "runs/experiment_v1/",
        }
        sau None la eroare
    """
    print("=" * 45)
    print("TRAINING PIPELINE - Anca")
    print("=" * 45)
    print(f"  Dataset:      {dataset_path}")
    print(f"  Epoci:        {epochs}")
    print(f"  Batch size:   {batch_size}")
    print(f"  Arhitectura:  {architecture}")
    print(f"  Output:       {output_dir}/{run_name}")
    print(f"  Device:       {device}")
    print()

    # Validare dataset
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"[train_model] EROARE: Nu gasesc dataset la '{dataset_path}'")
        print("              Ruleaza mai intai prepare_kitti.py!")
        return None

    try:
        # Construieste modelul
        model = build_model(architecture=architecture, pretrained=True)

        # Antrenare
        print("\n[train_model] Incep antrenarea...\n")
        results = model.train(
            data=str(dataset_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            lr0=learning_rate,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=min(3, epochs),
            device=device,
            project=output_dir,
            name=run_name,
            exist_ok=True,
            save=True,
            save_period=max(1, epochs // 5),
            val=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            fliplr=0.5,
            mosaic=1.0,
            flipud=0.0,
        )

        # Caile catre artefacte
        run_dir    = Path(output_dir) / run_name
        best_model = run_dir / "weights" / "best.pt"
        last_model = run_dir / "weights" / "last.pt"

        # Copiaza best.pt in models/ pentru Paul
        import shutil
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        best_model_copy = models_dir / "best.pt"
        if best_model.exists():
            shutil.copy(best_model, best_model_copy)
            print(f"\n  Model copiat pentru Paul: {best_model_copy}")

            # Copiaza best.pt in models/ pentru Paul
        import shutil
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        best_model_copy = models_dir / "best.pt"
        if best_model.exists():
            shutil.copy(best_model, best_model_copy)
            print(f"\n  Model copiat pentru Paul: {best_model_copy}")

        artifacts = {
            "best_model":  str(best_model_copy),
            "final_model": str(last_model),
            "run_dir":     str(run_dir),
        }

        print("\n" + "=" * 45)
        print("TRAINING FINALIZAT")
        print("=" * 45)
        print(f"  Best model:  {best_model}")
        print(f"  Last model:  {last_model}")
        print(f"  mAP@50:      {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")

        return artifacts

    except Exception as e:
        print(f"\n[train_model] EROARE in timpul training-ului: {e}")
        return None


# ============================================================
# 3. PARSE ARGS
# ============================================================

def parse_args():
    """
    Parseaza argumentele din linia de comanda.
    Permite rularea: python training.py --dataset ... --epochs 20
    """
    parser = argparse.ArgumentParser(
        description="Training YOLOv8 pe KITTI - Car/Pedestrian/Cyclist"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Calea catre dataset.yaml (ex: C:/date/kitti_yolo/dataset.yaml)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Numarul de epoci (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Marimea batch-ului (default: 4)"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Dimensiunea imaginilor (default: 640)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="yolov8n",
        choices=["yolov8n", "yolov8s", "yolov8m"],
        help="Arhitectura modelului (default: yolov8n)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs",
        help="Folderul pentru rezultate (default: runs)"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="experiment_v1",
        help="Numele experimentului (default: experiment_v1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: 'cpu' sau '0' pentru GPU (default: cpu)"
    )

    return parser.parse_args()


# ============================================================
# 4. MAIN
# ============================================================

def main():
    """
    Entrypoint CLI. Returneaza 0 la succes, 1 la eroare.
    """
    try:
        args = parse_args()

        artifacts = train_model(
            dataset_path=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            learning_rate=args.lr,
            architecture=args.architecture,
            output_dir=args.output_dir,
            run_name=args.run_name,
            device=args.device,
        )

        if artifacts is None:
            print("\n[main] Training esuat.")
            return 1

        print("\n[main] Succes!")
        return 0

    except Exception as e:
        print(f"\n[main] Eroare neasteptata: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())