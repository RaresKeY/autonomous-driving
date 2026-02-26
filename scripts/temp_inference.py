import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Temporary YOLO inference runner for local checkpoints."
    )
    parser.add_argument(
        "--model",
        default="models/best.pt",
        help="Path to YOLO checkpoint (default: models/best.pt)",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Image/video path, directory, webcam index, or URL accepted by Ultralytics",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for inference (default: cpu; use '0' for first GPU)",
    )
    parser.add_argument(
        "--project",
        default="runs/temp_inference",
        help="Output project directory for Ultralytics predictions",
    )
    parser.add_argument(
        "--name",
        default="predict",
        help="Run name inside project directory",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live inference window (if supported by environment)",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save YOLO-format prediction labels",
    )
    parser.add_argument(
        "--save-conf",
        action="store_true",
        help="Save confidence scores in label files (requires --save-txt)",
    )
    return parser.parse_args()


def main():
    try:
        args = parse_args()
        model_path = Path(args.model)

        if not model_path.exists():
            print(f"Model checkpoint not found: {model_path}")
            return 1

        from ultralytics import YOLO

        print(f"Loading model: {model_path}")
        model = YOLO(str(model_path))

        print(f"Running inference on source: {args.source}")
        results = model.predict(
            source=args.source,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name=args.name,
            exist_ok=True,
            save=True,
            show=args.show,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            verbose=True,
        )

        out_dir = Path(args.project) / args.name
        print(f"Inference complete. Output directory: {out_dir}")
        print(f"Results objects returned: {len(results)}")
        return 0

    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except Exception as exc:
        print(f"Inference failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
