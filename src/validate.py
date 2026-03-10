"""
============================================
Model Validation Script
============================================
Quickly check if your trained model achieves >90% mAP.

USAGE:
  python src/validate.py
  python src/validate.py --model models/best.pt
  python src/validate.py --model models/best.pt --data data/data.yaml
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def validate(model_path: str = "models/best.pt",
             data_yaml: str = "data/data.yaml",
             img_size: int = 640,
             conf: float = 0.4) -> dict:
    """
    Run validation on the trained model.

    Args:
        model_path: Path to best.pt weights
        data_yaml:  Path to data.yaml
        img_size:   Image size used during training
        conf:       Confidence threshold

    Returns:
        dict with mAP50, Precision, Recall
    """
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Train first with: python src/train.py")
        return {}

    print("=" * 55)
    print("  TRAFFIC DETECTION MODEL — VALIDATION")
    print("=" * 55)
    print(f"  Model: {model_file.name}")
    print(f"  Data:  {data_yaml}")
    print("=" * 55)

    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml,
        imgsz=img_size,
        conf=conf,
        iou=0.5,
        verbose=True,
    )

    map50    = metrics.box.map50
    map50_95 = metrics.box.map
    precision = metrics.box.mp
    recall    = metrics.box.mr

    print("\n" + "=" * 55)
    print("  RESULTS")
    print("=" * 55)
    print(f"  mAP@0.50:      {map50:.4f}  ({map50*100:.1f}%)  ← target ≥ 90%")
    print(f"  mAP@0.50:0.95: {map50_95:.4f}  ({map50_95*100:.1f}%)")
    print(f"  Precision:     {precision:.4f}  ({precision*100:.1f}%)")
    print(f"  Recall:        {recall:.4f}     ({recall*100:.1f}%)")
    print("=" * 55)

    if map50 >= 0.90:
        print(f"\n✅ Model PASSES accuracy target! mAP@50 = {map50*100:.1f}% ≥ 90%")
    else:
        gap = (0.90 - map50) * 100
        print(f"\n⚠️  Model is {gap:.1f}% below target ({map50*100:.1f}% vs 90%)")
        print("   Try:")
        print("     • More epochs:  python src/train.py --epochs 150")
        print("     • Bigger model: python src/train.py --model yolov8m.pt")
        print("     • More data:    add annotated images to your dataset")

    return {
        'map50': round(map50, 4),
        'map50_95': round(map50_95, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'passed': map50 >= 0.90,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Traffic Detection Model")
    parser.add_argument("--model", default="models/best.pt",
                        help="Path to trained model weights (.pt)")
    parser.add_argument("--data", default="data/data.yaml",
                        help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size (default: 640)")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Confidence threshold (default: 0.4)")

    args = parser.parse_args()
    validate(args.model, args.data, args.imgsz, args.conf)
