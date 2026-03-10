"""
============================================
Utility Functions
============================================
Common helpers used across the project.
"""

import os
import yaml
import json
from pathlib import Path
from datetime import datetime


def load_data_yaml(path="data/data.yaml"):
    """Load and return the YOLO data.yaml configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def get_class_names(data_yaml_path="data/data.yaml"):
    """Get class names from data.yaml."""
    config = load_data_yaml(data_yaml_path)
    names = config.get('names', {})
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]
    return list(names)


def count_dataset_images(data_yaml_path="data/data.yaml"):
    """Count images in train/val/test splits."""
    config = load_data_yaml(data_yaml_path)
    base = Path(config.get('path', ''))
    
    counts = {}
    for split in ['train', 'val', 'valid', 'test']:
        split_key = split
        if split_key not in config:
            if split == 'valid' and 'val' in config:
                split_key = 'val'
            elif split == 'val' and 'valid' in config:
                split_key = 'valid'
            else:
                continue
        
        img_dir = base / config[split_key]
        if img_dir.exists():
            image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            count = sum(1 for f in img_dir.iterdir() 
                       if f.suffix.lower() in image_exts)
            counts[split] = count
    
    return counts


def verify_dataset_structure(data_yaml_path="data/data.yaml"):
    """
    Verify the dataset is properly structured for training.
    Returns (ok: bool, messages: list).
    """
    messages = []
    ok = True
    
    if not os.path.exists(data_yaml_path):
        return False, [f"❌ data.yaml not found at: {data_yaml_path}"]
    
    config = load_data_yaml(data_yaml_path)
    base = Path(config.get('path', ''))
    
    if not base.exists():
        return False, [f"❌ Dataset path does not exist: {base}"]
    
    messages.append(f"✅ Dataset path: {base}")
    messages.append(f"✅ Classes ({config.get('nc', '?')}): {config.get('names', [])}")
    
    counts = count_dataset_images(data_yaml_path)
    for split, count in counts.items():
        if count > 0:
            messages.append(f"✅ {split}: {count} images")
        else:
            messages.append(f"⚠️ {split}: 0 images")
            ok = False
    
    if 'train' not in counts or counts.get('train', 0) == 0:
        ok = False
        messages.append("❌ No training images found!")
    
    return ok, messages


def format_detection_results(results_data):
    """Format detection results for logging/display."""
    lines = []
    lines.append(f"Frame #{results_data.get('frame', 0)}")
    lines.append(f"  Vehicles: {results_data.get('total_vehicles', 0)}")
    lines.append(f"  Density:  {results_data.get('total_density', 0):.1f}")
    
    counts = results_data.get('total_counts', {})
    if counts:
        breakdown = ", ".join(f"{k}:{v}" for k, v in counts.items() if v > 0)
        lines.append(f"  Breakdown: {breakdown}")
    
    emergency = results_data.get('emergency', {})
    if emergency.get('mode') == 'EMERGENCY':
        lines.append(f"  🚨 EMERGENCY: {emergency.get('green_lane', '?')}")
    
    return "\n".join(lines)


def save_results_log(results_data, log_dir="logs"):
    """Save detection results to JSON log file."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"detection_{timestamp}.json")
    
    with open(log_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    return log_file


def print_banner(text, width=60):
    """Print a formatted banner."""
    print("=" * width)
    print(f"  {text}")
    print("=" * width)
