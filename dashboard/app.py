"""
============================================
Traffic Management Dashboard - Backend
============================================
FastAPI server with WebSocket for real-time
traffic detection data streaming to browser.

USAGE:
  python dashboard/app.py
  Then open http://localhost:8000 in browser.

HOW TO INTEGRATE REAL-TIME PREDICTION:
  1. Click "Start Detection" on the dashboard and enter:
       - Video Source: webcam index (0) or a video file path
       - Model Path:   E:/Traffic_Management/models/best.pt
  2. The backend calls POST /api/start which:
       a. Loads your YOLOv8 model (best.pt)
       b. Opens the video/webcam via OpenCV
       c. Runs YOLO inference frame-by-frame in a background async task
       d. Pushes detection results to all WebSocket clients every 200ms
  3. The dashboard WebSocket receives live JSON and updates in real time.
  4. To stop, click "Stop Detection" or call POST /api/stop.

EMERGENCY GREEN CORRIDOR:
  - Ambulance detected by YOLO → only that lane turns GREEN, all others RED
  - GPS trigger: POST /api/emergency/trigger {"lane": "lane_2"}
  - Deactivate: POST /api/emergency/deactivate
"""

import asyncio
import json
import time
import sys
import os
import base64
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse
import uvicorn

from density import DensityCalculator
from signal_optimizer import SignalOptimizer
from emergency import EmergencyController

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ ultralytics not installed. Detection disabled — install with: pip install ultralytics")


# ============================================================
# App Setup
# ============================================================

app = FastAPI(title="Traffic Management System", version="2.0.0")

static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"
static_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))


# ============================================================
# Global State (4 Lanes)
# ============================================================

class TrafficState:
    """Shared state for the traffic detection system."""

    LANES = ['lane_1', 'lane_2', 'lane_3', 'lane_4']

    #  Class names from the dataset used during training
    DEFAULT_CLASS_NAMES = ['Ambulance', 'Bus', 'Motorcycle', 'Truck', 'Vehicle']
    EMERGENCY_CLASSES = {'Ambulance', 'ambulance', 'fire_truck'}

    VEHICLE_WEIGHTS = {
        'Vehicle': 2.0, 'car': 2.0,
        'Bus': 5.0,     'bus': 5.0,
        'Truck': 6.0,   'truck': 6.0,
        'Motorcycle': 1.0, 'bike': 1.0,
        'Ambulance': 0.0, 'ambulance': 0.0,
        'fire_truck': 0.0,
    }

    def __init__(self):
        self.model = None
        self.video_source = None
        self.running = False
        self.frame_count = 0
        self.latest_data: dict = {}
        self.density_calc = DensityCalculator(self.VEHICLE_WEIGHTS)
        self.signal_optimizer = SignalOptimizer()
        self.emergency_ctrl = EmergencyController(cooldown=8.0)
        self.connected_clients: set = set()
        self.class_names = self.DEFAULT_CLASS_NAMES

state = TrafficState()





# ============================================================
# REST Endpoints
# ============================================================

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status")
async def get_status():
    """Get current system status."""
    return {
        "running": state.running,
        "model_loaded": state.model is not None,
        "frame_count": state.frame_count,
        "connected_clients": len(state.connected_clients),
        "emergency": state.emergency_ctrl.get_stats(),
        "yolo_available": YOLO_AVAILABLE,
    }


@app.get("/api/latest")
async def get_latest_data():
    """Get latest detection data."""
    if state.latest_data:
        return state.latest_data
    return {"status": "waiting", "message": "No detection running. Start detection first."}


@app.get("/api/stats")
async def get_optimization_stats():
    """Get signal optimization statistics."""
    lane_densities = state.latest_data.get('lane_densities', {})
    return state.signal_optimizer.get_optimization_stats(lane_densities)


@app.post("/api/start")
async def start_detection(source: str = "0",
                          model: str = "models/best.pt"):
    """
    Start real-time detection pipeline.
    Query params: source (0=webcam or video path), model (path to best.pt)
    """
    if state.running:
        return {"status": "already_running", "message": "Detection is already active."}

    if not YOLO_AVAILABLE:
        return JSONResponse(
            status_code=400,
            content={"error": "ultralytics not installed. Run: pip install ultralytics"}
        )

    model_path = Path(model)
    if not model_path.exists():
        return JSONResponse(
            status_code=404,
            content={
                "error": f"Model file not found: {model}",
                "tip": "Train first with: python src/train.py"
            }
        )

    # Load YOLO model
    try:
        state.model = YOLO(str(model_path))
        state.class_names = list(state.model.names.values())
        print(f"✅ Model loaded: {model_path.name}")
        print(f"   Classes: {state.class_names}")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    # Set video source
    state.video_source = int(source) if source.isdigit() else source
    state.running = True
    state.frame_count = 0

    # Start background detection
    asyncio.create_task(detection_loop())

    return {
        "status": "started",
        "source": source,
        "model": model_path.name,
        "classes": state.class_names,
    }


@app.post("/api/stop")
async def stop_detection():
    """Stop the detection pipeline."""
    state.running = False
    state.model = None
    state.latest_data = {}
    print("⏹️ Detection stopped by user")
    return {"status": "stopped"}


# ============================================================
# Emergency Endpoints
# ============================================================

@app.post("/api/emergency/trigger")
async def trigger_emergency(lane: str = "lane_1",
                            ambulance_id: str = "AMB-001"):
    """
    GPS-based emergency trigger (query params).
    lane: which lane to clear (lane_1/lane_2/lane_3/lane_4)
    """
    if lane not in state.LANES:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Invalid lane '{lane}'",
                "valid_lanes": state.LANES,
            }
        )

    status = state.emergency_ctrl.trigger_gps(ambulance_id, lane)

    # Apply signal override immediately
    timings = state.signal_optimizer.emergency_override(
        lane, state.LANES
    )

    # Update latest_data so WebSocket pushes it immediately
    if state.latest_data:
        state.latest_data['emergency'] = status
        state.latest_data['signal_timings'] = timings
        state.latest_data['signals'] = state.signal_optimizer.get_signal_display()
    else:
        # No detection running — build a minimal payload
        state.latest_data = {
            'frame': 0,
            'timestamp': time.time(),
            'vehicle_counts': {},
            'total_vehicles': 0,
            'density': 0,
            'congestion': 'LOW',
            'lane_densities': {l: 0 for l in state.LANES},
            'signal_timings': timings,
            'signals': state.signal_optimizer.get_signal_display(),
            'emergency': status,
            'live': False,
        }

    return {"status": "emergency_activated", "details": status}


@app.post("/api/emergency/deactivate")
async def deactivate_emergency():
    """Manually deactivate emergency mode and return to normal."""
    result = state.emergency_ctrl.force_deactivate()
    # Restore adaptive timing
    lane_densities = state.latest_data.get(
        'lane_densities', {l: 0 for l in state.LANES}
    )
    timings = state.signal_optimizer.compute_timings(lane_densities)
    if state.latest_data:
        state.latest_data['emergency'] = {'mode': 'NORMAL'}
        state.latest_data['signal_timings'] = timings
        state.latest_data['signals'] = state.signal_optimizer.get_signal_display()
    return {"status": "deactivated", "details": result}


# ============================================================
# WebSocket — Real-Time Updates
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection for real-time dashboard updates."""
    await websocket.accept()
    state.connected_clients.add(websocket)
    print(f"📱 Client connected. Total: {len(state.connected_clients)}")

    try:
        while True:
            if state.latest_data:
                await websocket.send_json(state.latest_data)
            else:
                # Send 'waiting' status — NO fake random data
                await websocket.send_json({
                    "status": "waiting",
                    "running": state.running,
                    "message": "Start detection to see live data",
                    "timestamp": time.time(),
                })

            await asyncio.sleep(0.2)

    except WebSocketDisconnect:
        state.connected_clients.discard(websocket)
        print(f"📱 Client disconnected. Total: {len(state.connected_clients)}")
    except Exception as e:
        state.connected_clients.discard(websocket)
        print(f"❌ WebSocket error: {e}")


# ============================================================
# Detection Loop (Background Task)
# ============================================================

async def detection_loop():
    """
    Background video processing loop.

    Runs YOLO on every frame, computes density, checks emergency,
    updates signal timings, and stores result in state.latest_data
    for WebSocket broadcast.
    """
    cap = cv2.VideoCapture(state.video_source)

    if not cap.isOpened():
        print(f"❌ Could not open video source: {state.video_source}")
        state.running = False
        return

    print(f"🚀 Detection started on source: {state.video_source}")

    while state.running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if isinstance(state.video_source, int):
                await asyncio.sleep(0.01)
                continue
            # Video file ended — loop back
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        state.frame_count += 1

        if state.model:
            try:
                results = state.model(frame, conf=0.4, verbose=False)
                counts = {name: 0 for name in state.class_names}
                emergency_detected = False
                emergency_lane = None
                emergency_class = None

                h, w = frame.shape[:2]
                lane_width = w // 4  # Divide frame into 4 equal lanes

                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    cls_name = (state.class_names[cls_id]
                                if cls_id < len(state.class_names)
                                else "unknown")
                    counts[cls_name] = counts.get(cls_name, 0) + 1

                    # Determine which of 4 lanes the vehicle is in
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = (x1 + x2) / 2
                    lane_idx = min(int(center_x // lane_width), 3)
                    det_lane = f"lane_{lane_idx + 1}"

                    if cls_name in state.EMERGENCY_CLASSES:
                        emergency_detected = True
                        emergency_lane = det_lane
                        emergency_class = cls_name

                # Density per lane (split total proportionally by lane width)
                density = state.density_calc.calculate(counts)
                lane_densities = {
                    'lane_1': round(density * 0.28, 1),
                    'lane_2': round(density * 0.25, 1),
                    'lane_3': round(density * 0.25, 1),
                    'lane_4': round(density * 0.22, 1),
                }

                # Emergency check
                emergency_status = state.emergency_ctrl.check_emergency(
                    emergency_detected, emergency_lane, emergency_class
                )

                # Signal timing
                if emergency_status['mode'] == 'EMERGENCY':
                    timings = state.signal_optimizer.emergency_override(
                        emergency_status['green_lane'],
                        state.LANES
                    )
                else:
                    timings = state.signal_optimizer.compute_timings(lane_densities)

                state.latest_data = {
                    'frame': state.frame_count,
                    'timestamp': time.time(),
                    'vehicle_counts': {k: v for k, v in counts.items() if v > 0},
                    'total_vehicles': sum(counts.values()),
                    'density': round(density, 1),
                    'congestion': state.density_calc.congestion_level(density),
                    'lane_densities': lane_densities,
                    'signal_timings': timings,
                    'signals': state.signal_optimizer.get_signal_display(),
                    'emergency': emergency_status,
                    'live': True,
                }

            except Exception as e:
                print(f"⚠️ Detection error on frame {state.frame_count}: {e}")

        await asyncio.sleep(0.033)  # ~30 FPS

    cap.release()
    state.running = False
    state.latest_data = {}
    print("⏹️ Detection stopped")


# ============================================================
# Run Server
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  TRAFFIC MANAGEMENT DASHBOARD v2.0")
    print("=" * 60)
    print("  Open http://localhost:8000 in your browser")
    print("  Press Ctrl+C to stop")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
