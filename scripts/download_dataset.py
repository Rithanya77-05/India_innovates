"""
============================================
Dataset Download Helper
============================================
Downloads a pre-annotated traffic vehicle detection dataset
from Roboflow Universe in YOLOv8 format.

INSTRUCTIONS:
1. Go to https://universe.roboflow.com
2. Sign up for FREE (Google login works)
3. Search for "vehicle detection" or use one of these direct links:
   
   RECOMMENDED DATASETS:
   ─────────────────────
   A) Traffic & Vehicle Detection (includes ambulance):
      https://universe.roboflow.com/yolo-trials-acaaf/vehicle-detection-fkxiy
      → Classes: car, truck, bus, ambulance, motorcycle
      → ~2000+ images, pre-annotated
   
   B) Vehicle Dataset (6 classes):
      https://universe.roboflow.com/search?q=traffic+vehicle+detection+yolov8
      → Search and pick one with car/bus/truck/ambulance
   
   C) Emergency Vehicle Detection:
      https://universe.roboflow.com/search?q=ambulance+detection
      → Specifically for ambulance/fire truck detection

4. On the dataset page, click "Download Dataset"
5. Choose format: "YOLOv8"
6. Choose "download zip to computer" OR "show download code"
7. If you chose "show download code", copy the API key and workspace info below

ALTERNATIVE: Use the Roboflow Python API (Option 2 below)
"""

import os
import sys
import zipfile
import shutil


# ============================================================
# OPTION 1: Manual Download (EASIEST - No API key needed)
# ============================================================
def manual_download_instructions():
    """
    Print step-by-step instructions for manual dataset download.
    """
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          DATASET DOWNLOAD - STEP BY STEP GUIDE                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  OPTION A: Download from ROBOFLOW (Recommended)                  ║
║  ───────────────────────────────────────────────                  ║
║  1. Open browser → https://universe.roboflow.com                 ║
║  2. Sign up FREE (use Google login)                              ║
║  3. Search: "vehicle detection" or "traffic detection"           ║
║  4. Pick a dataset with these classes:                           ║
║     car, bus, truck, bike, ambulance                             ║
║                                                                  ║
║  RECOMMENDED:                                                    ║
║  → https://universe.roboflow.com/yolo-trials-acaaf/              ║
║    vehicle-detection-fkxiy                                       ║
║                                                                  ║
║  5. Click "Download Dataset"                                     ║
║  6. Format: select "YOLOv8"                                      ║
║  7. Click "download zip to computer"                             ║
║  8. Extract the ZIP file                                         ║
║  9. You will get folders like:                                   ║
║                                                                  ║
║     downloaded_dataset/                                          ║
║     ├── train/                                                   ║
║     │   ├── images/    (training images)                         ║
║     │   └── labels/    (YOLO .txt annotations)                  ║
║     ├── valid/                                                   ║
║     │   ├── images/                                              ║
║     │   └── labels/                                              ║
║     ├── test/                                                    ║
║     │   ├── images/                                              ║
║     │   └── labels/                                              ║
║     └── data.yaml                                                ║
║                                                                  ║
║  10. Copy ALL these folders into:                                ║
║      E:\\Traffic_Management\\data\\dataset\\                      ║
║                                                                  ║
║      So final structure is:                                      ║
║      E:\\Traffic_Management\\data\\dataset\\                      ║
║      ├── train/                                                  ║
║      │   ├── images/                                             ║
║      │   └── labels/                                             ║
║      ├── valid/                                                  ║
║      │   ├── images/                                             ║
║      │   └── labels/                                             ║
║      └── test/                                                   ║
║          ├── images/                                             ║
║          └── labels/                                             ║
║                                                                  ║
║  11. Update data/data.yaml with correct class names              ║
║      from the dataset's own data.yaml                            ║
║                                                                  ║
║                                                                  ║
║  OPTION B: Download from KAGGLE                                  ║
║  ──────────────────────────────                                  ║
║  1. Go to https://www.kaggle.com/datasets                        ║
║  2. Search: "vehicle detection YOLO"                             ║
║                                                                  ║
║  RECOMMENDED Kaggle datasets:                                    ║
║  → "Vehicle Dataset for YOLO" (3000 images, 6 classes)           ║
║    https://www.kaggle.com/search?q=vehicle+dataset+yolo          ║
║                                                                  ║
║  3. Download & extract                                           ║
║  4. If it comes as one big folder of images + labels:            ║
║     Run: python scripts/split_dataset.py                         ║
║     to split into train/valid/test                               ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)


# ============================================================
# OPTION 2: Automated Download via Roboflow API
# ============================================================
def download_from_roboflow(api_key, workspace, project, version_num=1):
    """
    Download dataset automatically using Roboflow API.
    
    HOW TO GET THESE VALUES:
    1. Sign up at https://roboflow.com (free)
    2. Go to Settings → API Key → copy your key
    3. On the dataset page, click "Download Dataset" → "Show Download Code"
    4. Copy the workspace name, project name, and version number
    
    Example:
        api_key = "YOUR_API_KEY_HERE"
        workspace = "yolo-trials-acaaf"
        project = "vehicle-detection-fkxiy"
        version_num = 1
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("❌ Roboflow not installed. Run: pip install roboflow")
        return
    
    print(f"📥 Downloading dataset from Roboflow...")
    print(f"   Workspace: {workspace}")
    print(f"   Project:   {project}")
    print(f"   Version:   {version_num}")
    
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    version = proj.version(version_num)
    
    # Download in YOLOv8 format directly into our data folder
    dataset = version.download(
        "yolov8",
        location="E:/Traffic_Management/data/dataset"
    )
    
    print(f"✅ Dataset downloaded to: E:/Traffic_Management/data/dataset")
    print(f"   Now update data/data.yaml if class names differ!")
    return dataset


# ============================================================
# OPTION 3: Organize an already-downloaded dataset
# ============================================================
def organize_downloaded_dataset(source_path, dest_path="E:/Traffic_Management/data/dataset"):
    """
    If you downloaded a dataset ZIP and extracted it somewhere,
    this function copies it into the correct project structure.
    
    Handles these common downloaded structures:
    
    Structure A (Roboflow-style):
        source/
        ├── train/images/ + train/labels/
        ├── valid/images/ + valid/labels/
        └── test/images/  + test/labels/
    
    Structure B (Flat folder):
        source/
        ├── images/   (all images in one folder)
        └── labels/   (all labels in one folder)
    
    Structure C (Mixed):
        source/
        ├── img001.jpg
        ├── img001.txt
        ├── img002.jpg
        └── img002.txt
    """
    
    if not os.path.exists(source_path):
        print(f"❌ Source path not found: {source_path}")
        return
    
    os.makedirs(dest_path, exist_ok=True)
    
    # Check Structure A: already has train/valid/test folders
    if os.path.isdir(os.path.join(source_path, "train")):
        print("📁 Detected Structure A (Roboflow-style with train/valid/test)")
        for split in ["train", "valid", "val", "test"]:
            src_split = os.path.join(source_path, split)
            if os.path.isdir(src_split):
                dest_split_name = "valid" if split == "val" else split
                dest_split = os.path.join(dest_path, dest_split_name)
                if os.path.exists(dest_split):
                    shutil.rmtree(dest_split)
                shutil.copytree(src_split, dest_split)
                
                # Count files
                img_dir = os.path.join(dest_split, "images")
                if os.path.isdir(img_dir):
                    count = len([f for f in os.listdir(img_dir) 
                               if f.endswith(('.jpg','.jpeg','.png'))])
                    print(f"   ✅ {dest_split_name}: {count} images")
        
        # Copy data.yaml if it exists in source
        src_yaml = os.path.join(source_path, "data.yaml")
        if os.path.exists(src_yaml):
            shutil.copy2(src_yaml, os.path.join(dest_path, "data.yaml"))
            print("   ✅ Copied data.yaml")
        
        print(f"\n✅ Dataset organized at: {dest_path}")
        return
    
    # Check Structure B: has images/ and labels/ folders
    if os.path.isdir(os.path.join(source_path, "images")):
        print("📁 Detected Structure B (flat images/labels folders)")
        print("   → Will split into train/valid/test (70/20/10)")
        _split_flat_dataset(
            os.path.join(source_path, "images"),
            os.path.join(source_path, "labels"),
            dest_path
        )
        return
    
    # Check Structure C: mixed (images and labels in same folder)
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [f for f in os.listdir(source_path) 
              if os.path.splitext(f)[1].lower() in image_exts]
    if images:
        print("📁 Detected Structure C (mixed images and labels)")
        print("   → Will separate and split into train/valid/test (70/20/10)")
        
        # Create temp folders
        tmp_images = os.path.join(source_path, "_tmp_images")
        tmp_labels = os.path.join(source_path, "_tmp_labels")
        os.makedirs(tmp_images, exist_ok=True)
        os.makedirs(tmp_labels, exist_ok=True)
        
        for img in images:
            shutil.copy2(os.path.join(source_path, img), tmp_images)
            label = os.path.splitext(img)[0] + ".txt"
            label_path = os.path.join(source_path, label)
            if os.path.exists(label_path):
                shutil.copy2(label_path, tmp_labels)
        
        _split_flat_dataset(tmp_images, tmp_labels, dest_path)
        
        # Cleanup temp
        shutil.rmtree(tmp_images)
        shutil.rmtree(tmp_labels)
        return
    
    print("❌ Could not detect dataset structure. Please organize manually.")


def _split_flat_dataset(images_dir, labels_dir, dest_path, ratios=(0.7, 0.2, 0.1)):
    """Split a flat images/labels folder into train/valid/test."""
    import random
    
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = sorted([f for f in os.listdir(images_dir) 
                     if os.path.splitext(f)[1].lower() in image_exts])
    
    random.shuffle(images)
    n = len(images)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])
    
    splits = {
        'train': images[:train_end],
        'valid': images[train_end:val_end],
        'test': images[val_end:]
    }
    
    for split_name, split_files in splits.items():
        img_dest = os.path.join(dest_path, split_name, "images")
        lbl_dest = os.path.join(dest_path, split_name, "labels")
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(lbl_dest, exist_ok=True)
        
        for f in split_files:
            shutil.copy2(os.path.join(images_dir, f), img_dest)
            label = os.path.splitext(f)[0] + ".txt"
            label_path = os.path.join(labels_dir, label)
            if os.path.exists(label_path):
                shutil.copy2(label_path, lbl_dest)
        
        print(f"   ✅ {split_name}: {len(split_files)} images")


# ============================================================
# MAIN - Run this script for instructions
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  TRAFFIC MANAGEMENT - DATASET SETUP")
    print("=" * 60)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        # Usage: python download_dataset.py --api YOUR_KEY WORKSPACE PROJECT VERSION
        if len(sys.argv) < 5:
            print("Usage: python download_dataset.py --api API_KEY WORKSPACE PROJECT [VERSION]")
            print("Example: python download_dataset.py --api abc123 yolo-trials vehicle-detection 1")
            sys.exit(1)
        
        api_key = sys.argv[2]
        workspace = sys.argv[3]
        project = sys.argv[4]
        version = int(sys.argv[5]) if len(sys.argv) > 5 else 1
        download_from_roboflow(api_key, workspace, project, version)
    
    elif len(sys.argv) > 1 and sys.argv[1] == "--organize":
        # Usage: python download_dataset.py --organize /path/to/extracted/dataset
        if len(sys.argv) < 3:
            print("Usage: python download_dataset.py --organize /path/to/extracted/folder")
            sys.exit(1)
        organize_downloaded_dataset(sys.argv[2])
    
    else:
        manual_download_instructions()
        print("\nOTHER OPTIONS:")
        print("─" * 40)
        print("  Auto-download via Roboflow API:")
        print("    python scripts/download_dataset.py --api YOUR_KEY WORKSPACE PROJECT")
        print()
        print("  Organize an already-downloaded dataset:")
        print("    python scripts/download_dataset.py --organize C:/path/to/extracted/zip")
