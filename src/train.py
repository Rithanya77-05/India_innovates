
"""
============================================
YOLOv8 Traffic Model - Training Script
============================================
Trains (fine-tunes) YOLOv8 on your traffic vehicle dataset.

PREREQUISITES:
  1. Dataset is downloaded and placed in data/dataset/
  2. data/data.yaml is updated with correct paths and class names
  3. Dependencies installed: pip install -r requirements.txt

USAGE:
  python src/train.py
  python src/train.py --model yolov8s.pt --epochs 100 --batch 8
  
NOTE FOR AMD GPU USERS:
  YOLOv8 uses PyTorch which requires CUDA (NVIDIA) for GPU.
  With AMD Radeon, training will run on CPU automatically.
  Use yolov8n.pt (nano) for fastest CPU training (~2-3 hours for 50 epochs).
  If you have access to Google Colab (free GPU), upload your dataset there.
"""

import argparse
import os
import shutil
from pathlib import Path
from ultralytics import YOLO


def check_dataset():
    """Verify the dataset is in place before training."""
    data_yaml = Path("data/data.yaml")
    if not data_yaml.exists():
        print("❌ data/data.yaml not found!")
        print("   Run: python scripts/download_dataset.py")
        return False
    
    # Read data.yaml to check paths
    import yaml
    with open(data_yaml) as f:
        config = yaml.safe_load(f)
    
    dataset_path = Path(config.get("path", ""))
    train_path = dataset_path / config.get("train", "train/images")
    
    if not train_path.exists():
        print(f"❌ Training images not found at: {train_path}")
        print(f"   Expected dataset at: {dataset_path}")
        print(f"\n   SOLUTIONS:")
        print(f"   1. Download dataset: python scripts/download_dataset.py")
        print(f"   2. Check path in data/data.yaml")
        return False
    
    # Count training images
    img_count = len(list(train_path.glob("*.jpg")) + 
                     list(train_path.glob("*.png")) +
                     list(train_path.glob("*.jpeg")))
    print(f"✅ Found {img_count} training images at: {train_path}")
    return True


def train(model_name="yolov8s.pt", epochs=100, batch_size=16, img_size=640, device="cpu"):
    """
    Train YOLOv8 on the traffic dataset.
    
    Args:
        model_name: Pretrained model to start from
                    'yolov8n.pt' = Nano (6MB, fastest, recommended for CPU)
                    'yolov8s.pt' = Small (22MB, better accuracy)
                    'yolov8m.pt' = Medium (52MB, best balance - needs GPU)
        epochs: Number of training epochs (50 for hackathon, 100+ for production)
        batch_size: Images per batch (reduce if you get memory errors)
        img_size: Image resize dimension (640 standard)
        device: 'cpu' for AMD/Intel, '0' for NVIDIA GPU
    """
    
    print("=" * 60)
    print("  TRAFFIC VEHICLE DETECTION - MODEL TRAINING")
    print("=" * 60)
    print(f"  Model:      {model_name}")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}")
    print(f"  Device:     {device}")
    print("=" * 60)
    
    # Verify dataset exists
    if not check_dataset():
        return
    
    # 1. Load pretrained model (auto-downloads if not present)
    print(f"\n📦 Loading pretrained model: {model_name}")
    model = YOLO(model_name)
    
    # 2. Train on traffic dataset
    print(f"\n🚀 Starting training for {epochs} epochs...")
    print(f"   This will take approximately:")
    if device == "cpu":
        est_time = epochs * 3  # ~3 min per epoch on CPU for nano model
        print(f"   ~{est_time // 60}h {est_time % 60}m on CPU (estimated)")
    else:
        est_time = epochs * 0.5  # ~30sec per epoch on GPU
        print(f"   ~{est_time:.0f} minutes on GPU (estimated)")
    
    results = model.train(
        data="data/data.yaml",
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project="runs/train",
        name="traffic_model",

        # Optimizer — AdamW converges faster than SGD
        optimizer="AdamW",
        lr0=0.001,               # Lower initial LR for fine-tuning (key for accuracy)
        lrf=0.01,                # Final LR factor (lr_final = lr0 * lrf)
        cos_lr=True,             # Cosine LR scheduler — smoother decay, better mAP
        warmup_epochs=5,         # Warmup prevents early instability
        warmup_momentum=0.8,

        # Augmentation — prevents overfitting, greatly improves generalisation
        hsv_h=0.015,             # Hue jitter
        hsv_s=0.7,               # Saturation jitter
        hsv_v=0.4,               # Brightness jitter
        degrees=10.0,            # Rotation augmentation
        scale=0.5,               # Random scale (zoom in/out)
        shear=2.0,               # Shear augmentation
        perspective=0.0005,      # Perspective warp
        flipud=0.0,              # No vertical flip
        fliplr=0.5,              # Horizontal flip
        mosaic=1.0,              # Mosaic (combines 4 images) — critical for density
        mixup=0.15,              # MixUp blending
        copy_paste=0.1,          # Copy-paste augmentation for small objects
        auto_augment="randaugment",  # AutoAugment policy — proven +2-5% mAP
        erasing=0.4,             # Random erasing (occlusion simulation)
        close_mosaic=10,         # Disable mosaic for last 10 epochs (stabilises)

        # Training behavior
        patience=20,             # Don't stop early — let model fully converge
        label_smoothing=0.1,     # Label smoothing reduces overconfidence
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )
    
    # 3. Validate the model
    print("\n📊 Running validation...")
    metrics = model.val()
    
    print("\n" + "=" * 60)
    print("  TRAINING RESULTS")
    print("=" * 60)
    print(f"  mAP50:       {metrics.box.map50:.4f}  (target > 0.85)")
    print(f"  mAP50-95:    {metrics.box.map:.4f}")
    print(f"  Precision:   {metrics.box.mp:.4f}  (target > 0.80)")
    print(f"  Recall:      {metrics.box.mr:.4f}  (target > 0.80)")
    print("=" * 60)
    
    # 4. Copy best weights to models/ folder
    best_weights = Path("runs/train/traffic_model/weights/best.pt")
    if best_weights.exists():
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        dest = models_dir / "best.pt"
        shutil.copy2(best_weights, dest)
        print(f"\n✅ Best model saved to: {dest}")
    
    # 5. Print next steps
    print("\n🎯 NEXT STEPS:")
    print("   1. Check training plots at: runs/train/traffic_model/")
    print("   2. Test detection: python src/detect.py")
    print("   3. If mAP < 0.85, try:")
    print("      - More epochs: python src/train.py --epochs 100")
    print("      - Bigger model: python src/train.py --model yolov8s.pt")
    print("      - More data: add more annotated images")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 Traffic Detection Model")
    parser.add_argument("--model", default="yolov8s.pt",
                       help="Pretrained model — use yolov8s.pt for ~90%+ mAP (yolov8n/s/m/l/x.pt)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of epochs (default: 100 for 90%+ mAP)")
    parser.add_argument("--batch", type=int, default=16, 
                       help="Batch size (reduce to 8 or 4 if memory error)")
    parser.add_argument("--imgsz", type=int, default=640, 
                       help="Image size (default: 640)")
    parser.add_argument("--device", default="cpu",
                       help="Device: 'cpu' for AMD, '0' for NVIDIA GPU")
    
    args = parser.parse_args()
    train(args.model, args.epochs, args.batch, args.imgsz, args.device)
