# 🚦 Dynamic AI Traffic Flow Optimizer & Emergency Grid

> **AI-powered real-time traffic management system** that uses computer vision (YOLOv8) to detect vehicles, compute lane-wise traffic density, dynamically adjust signal timings, and activate emergency green corridors for ambulances/fire trucks.

---

## 📁 Project Structure

```
Traffic_Management/
├── data/
│   ├── data.yaml              ← YOLO dataset config (edit class names here)
│   └── dataset/               ← PUT YOUR DOWNLOADED DATASET HERE
│       ├── train/images/ + labels/
│       ├── valid/images/ + labels/
│       └── test/images/  + labels/
│
├── models/
│   └── best.pt                ← Your trained model (created after training)
│
├── src/
│   ├── train.py               ← Model training script
│   ├── detect.py              ← Real-time detection + counting
│   ├── density.py             ← Density score calculator
│   ├── signal_optimizer.py    ← Adaptive signal timing
│   ├── emergency.py           ← Emergency green corridor
│   └── utils.py               ← Helper utilities
│
├── dashboard/
│   ├── app.py                 ← FastAPI server (WebSocket)
│   ├── templates/index.html   ← Dashboard UI
│   └── static/
│       ├── style.css          ← Dashboard styling
│       └── script.js          ← Real-time updates
│
├── scripts/
│   └── download_dataset.py    ← Dataset download helper
│
├── runs/                      ← Training outputs (auto-created)
├── requirements.txt
└── README.md
```

---

## 🚀 Complete A-Z Setup Guide

### Step 1: Install Python Dependencies

```bash
cd E:\Traffic_Management
pip install -r requirements.txt
```

> **AMD GPU Users:** YOLOv8 uses PyTorch with CUDA (NVIDIA only). Training will automatically run on CPU. Use `yolov8n.pt` (nano model) for fastest CPU training.

---

### Step 2: Download the Dataset

Run the helper script to see download instructions:

```bash
python scripts/download_dataset.py
```

#### Option A: Roboflow (⭐ RECOMMENDED — Fastest)

1. Go to **https://universe.roboflow.com**
2. Sign up **FREE** (Google login works)
3. Search for **"vehicle detection"** or **"traffic detection"**
4. **Recommended datasets:**
   - Search: `vehicle detection yolov8`
   - Look for datasets with classes: car, bus, truck, bike, ambulance
5. Click **"Download Dataset"**
6. Select format: **YOLOv8**
7. Click **"Download ZIP to computer"**
8. Extract the ZIP — you'll get folders like:
   ```
   extracted_folder/
   ├── train/
   │   ├── images/    ← training images (.jpg)
   │   └── labels/    ← annotation files (.txt)
   ├── valid/
   │   ├── images/
   │   └── labels/
   ├── test/
   │   ├── images/
   │   └── labels/
   └── data.yaml      ← class definitions
   ```

#### Option B: Kaggle

1. Go to **https://www.kaggle.com/datasets**
2. Search: **"Vehicle Dataset for YOLO"** (3000 images, 6 classes)
3. Download and extract

---

### Step 3: Place the Dataset

Copy the extracted dataset folders into `data/dataset/`:

```
E:\Traffic_Management\data\dataset\
├── train/
│   ├── images/     ← All training .jpg/.png files go here
│   └── labels/     ← All training .txt annotation files go here
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**If your download has a different structure**, use the auto-organizer:

```bash
python scripts/download_dataset.py --organize "C:\path\to\extracted\folder"
```

This handles 3 common structures:
- **Structure A:** Already has train/valid/test → copies directly
- **Structure B:** Single images/ + labels/ folder → auto-splits 70/20/10
- **Structure C:** Mixed images and labels together → separates and splits

---

### Step 4: Update data.yaml (If Needed)

Open `data/data.yaml` and update the class names to match your dataset:

```yaml
path: E:/Traffic_Management/data/dataset
train: train/images
val: valid/images
test: test/images

nc: 6
names:
  0: car
  1: bus
  2: truck
  3: bike
  4: ambulance
  5: fire_truck
```

> **Important:** The class names and `nc` (number of classes) MUST match what's in your dataset. Check the `data.yaml` file that came with your downloaded dataset.

---

### Step 5: Train the Model

```bash
python src/train.py
```

**Options:**
```bash
python src/train.py --model yolov8n.pt --epochs 50 --batch 16 --device cpu
```

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Base model (`yolov8n.pt` = nano, `yolov8s.pt` = small) | `yolov8n.pt` |
| `--epochs` | Training iterations (50 for hackathon, 100+ production) | `50` |
| `--batch` | Batch size (reduce to 8 or 4 if memory error) | `16` |
| `--device` | `cpu` for AMD, `0` for NVIDIA GPU | `cpu` |

**Training time estimates:**
- **CPU (AMD):** ~2-3 hours for 50 epochs with yolov8n
- **GPU (NVIDIA):** ~15-30 minutes for 50 epochs

**Output:** `runs/train/traffic_model/weights/best.pt` → auto-copied to `models/best.pt`

**Target metrics:**
- mAP50 > 0.85
- Precision > 0.80
- Recall > 0.80

---

### Step 6: Test Detection

Run on a video file:
```bash
python src/detect.py --source path/to/traffic_video.mp4
```

Run on webcam:
```bash
python src/detect.py --source 0
```

With lane counting:
```bash
python src/detect.py --source video.mp4 --lanes
```

Save annotated output:
```bash
python src/detect.py --source video.mp4 --output output.mp4
```

**Controls:**
- Press `q` to quit
- Press `p` to pause

---

### Step 7: Launch Dashboard

```bash
python dashboard/app.py
```

Open **http://localhost:8000** in your browser.

The dashboard shows:
- 🚗 Live vehicle counts
- 📊 Lane density bars (real-time)
- 🚦 Animated traffic signals with adaptive timers
- 📈 Density history graph
- 🚨 Emergency alert banner

> The dashboard works in **demo mode** even without a camera/model — it generates simulated data so you can test the UI immediately.

---

## 🔑 How Each Part Works

### Vehicle Detection (YOLOv8)
- Detects: car, bus, truck, bike, ambulance, fire truck
- Runs on each video frame (30 FPS target)
- Returns bounding boxes + class + confidence

### Density Scoring
- Each vehicle type has a weight: bike=1, car=2, bus=5, truck=6
- **Density = Σ(count × weight)** per lane
- Congestion levels: LOW (<15), MODERATE (15-30), HIGH (30-50), CRITICAL (>50)

### Adaptive Signal Timing
- **Green time = (lane density / total density) × 120 seconds**
- Clamped to min 10s, max 60s per lane
- Updates every frame based on real-time density

### Emergency Green Corridor
- When ambulance/fire truck detected by YOLO:
  - That lane immediately gets GREEN
  - All other lanes get RED
  - Stays active until vehicle not seen for 5 seconds
  - Then returns to normal adaptive mode

---

## 🏆 Hackathon Presentation Metrics

Add these to your slides:
- ✅ **Reduces average waiting time by X%** (use optimization stats endpoint)
- ✅ **Reduces fuel wastage** (less idle time = less emissions)
- ✅ **Emergency response time reduced** (green corridor)
- ✅ **Scalable to smart cities** (just add more cameras)
- ✅ **Real-time: 30+ FPS processing**

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `pip install` fails | Use `pip install --upgrade pip` first |
| Out of memory during training | Reduce `--batch` to 8 or 4 |
| mAP is low | More epochs, bigger model, more data |
| `ModuleNotFoundError: ultralytics` | Run `pip install ultralytics` |
| Webcam not opening | Try `--source 1` or check camera permissions |
| Dashboard not loading | Check http://localhost:8000, ensure `uvicorn` is running |
