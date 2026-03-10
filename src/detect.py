"""
============================================
Traffic Detection & Vehicle Counting
============================================
Runs YOLOv8 inference on video/webcam/images and counts vehicles
per lane, calculates density scores, and detects emergency vehicles.

USAGE:
  python src/detect.py                                    # Webcam
  python src/detect.py --source path/to/video.mp4         # Video file
  python src/detect.py --source path/to/images/           # Image folder
  python src/detect.py --source 0                         # Webcam (device 0)
"""

import argparse
import json
import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from density import DensityCalculator
from signal_optimizer import SignalOptimizer
from emergency import EmergencyController


# ============================================================
# Configuration
# ============================================================

# Class names matching your downloaded dataset
DEFAULT_CLASSES = ['car', 'motorcycle', 'bus', 'truck', 'bicycle', 'person']

# Vehicle weights for density scoring
VEHICLE_WEIGHTS = {
    'Vehicle': 2.0,       # Cars and general vehicles
    'Bus': 5.0,
    'Truck': 6.0,
    'Motorcycle': 1.0,
    'Ambulance': 0.0,     # Not counted in normal density (emergency)
}

# Emergency vehicle class names
EMERGENCY_CLASSES = {'Ambulance'}

# Colors for visualization (BGR format)
CLASS_COLORS = {
    'Vehicle': (0, 255, 0),      # Green
    'Bus': (255, 165, 0),        # Orange
    'Truck': (0, 0, 255),        # Red
    'Motorcycle': (255, 255, 0), # Cyan
    'Ambulance': (0, 0, 255),    # Red (emergency)
}


# ============================================================
# Lane Zone Configuration
# ============================================================
# Define lane zones as polygon regions on the video frame.
# These coordinates depend on your camera angle/position.
# UPDATE THESE to match your actual video/camera setup.
#
# Format: list of (x, y) points forming a polygon
# Tip: Run detect.py once, pause the video, note the pixel
#      coordinates for each lane boundary.

LANE_ZONES = {
    'lane_1': np.array([[0,   300], [160, 300], [160, 700], [0,   700]]),
    'lane_2': np.array([[160, 300], [320, 300], [320, 700], [160, 700]]),
    'lane_3': np.array([[320, 300], [480, 300], [480, 700], [320, 700]]),
    'lane_4': np.array([[480, 300], [640, 300], [640, 700], [480, 700]]),
}

# Set to None to disable lane-based counting (count all vehicles together)
# LANE_ZONES = None


class TrafficDetector:
    """
    Main traffic detection and analysis system.
    
    Combines:
    - YOLOv8 object detection
    - Per-lane vehicle counting
    - Density score calculation
    - Emergency vehicle detection
    - Adaptive signal timing
    """
    
    def __init__(self, model_path="models/best.pt",
                 class_names=None, confidence=0.4, lane_zones=None):
        """
        Args:
            model_path: Path to trained YOLOv8 weights
            class_names: List of class names (auto-detected from model if None)
            confidence: Detection confidence threshold (0-1)
            lane_zones: Dict of lane name → polygon numpy array (None = no lanes)
        """
        print(f"📦 Loading model: {model_path}")
        self.model = YOLO(model_path)
        
        # Get class names from model or use provided ones
        if class_names:
            self.class_names = class_names
        else:
            self.class_names = list(self.model.names.values())
        
        print(f"   Classes: {self.class_names}")
        
        self.confidence = confidence
        self.lane_zones = lane_zones
        
        # Initialize sub-systems
        self.density_calc = DensityCalculator(VEHICLE_WEIGHTS)
        self.signal_optimizer = SignalOptimizer()
        self.emergency_ctrl = EmergencyController()
        
        # Stats tracking
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
    
    def detect_frame(self, frame):
        """
        Process a single frame:
        1. Run YOLO detection
        2. Count vehicles per lane
        3. Compute density
        4. Check for emergency vehicles
        5. Calculate signal timings
        
        Returns:
            annotated_frame: Frame with bounding boxes drawn
            results_data: Dict with all detection data
        """
        self.frame_count += 1
        
        # 1. Run YOLO detection
        results = self.model(frame, conf=self.confidence, iou=0.4, max_det=1000, verbose=False)
        detections = results[0]
        
        # 2. Parse detections
        vehicles = []
        lane_counts = {}
        if self.lane_zones:
            lane_counts = {lane: {} for lane in self.lane_zones}
        total_counts = {name: 0 for name in self.class_names}
        emergency_detected = False
        emergency_lane = None
        emergency_class = None
        
        for box in detections.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            vehicle = {
                'class': cls_name,
                'confidence': conf,
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'center': [float(center_x), float(center_y)],
            }
            
            # Determine lane
            if self.lane_zones:
                lane = self._get_lane(center_x, center_y)
                vehicle['lane'] = lane
                if lane and lane in lane_counts:
                    lane_counts[lane][cls_name] = lane_counts[lane].get(cls_name, 0) + 1
            
            vehicles.append(vehicle)
            total_counts[cls_name] = total_counts.get(cls_name, 0) + 1
            
            # Check for emergency vehicles
            if cls_name in EMERGENCY_CLASSES:
                emergency_detected = True
                emergency_lane = vehicle.get('lane', 'unknown')
                emergency_class = cls_name
        
        # 3. Compute density scores
        if self.lane_zones:
            lane_densities = {}
            for lane, counts in lane_counts.items():
                lane_densities[lane] = self.density_calc.calculate(counts)
            total_density = sum(lane_densities.values())
        else:
            total_density = self.density_calc.calculate(total_counts)
            lane_densities = {'all': total_density}
        
        # 4. Check emergency override
        emergency_status = self.emergency_ctrl.check_emergency(
            emergency_detected, emergency_lane, emergency_class
        )
        
        # 5. Calculate signal timings
        if emergency_status['mode'] == 'EMERGENCY':
            signal_timings = self.signal_optimizer.emergency_override(
                emergency_status['green_lane'], 
                list(lane_densities.keys())
            )
        else:
            signal_timings = self.signal_optimizer.compute_timings(lane_densities)
        
        # 6. Calculate FPS
        current_time = time.time()
        if self.frame_count % 10 == 0:
            self.fps = 10 / (current_time - self.last_time)
            self.last_time = current_time
        
        # 7. Draw annotations
        annotated_frame = self._draw_annotations(
            frame, detections, vehicles, lane_densities,
            signal_timings, emergency_status
        )
        
        # Build results
        results_data = {
            'frame': self.frame_count,
            'fps': round(self.fps, 1),
            'vehicles': vehicles,
            'total_counts': total_counts,
            'lane_densities': lane_densities,
            'total_density': total_density,
            'signal_timings': signal_timings,
            'emergency': emergency_status,
            'total_vehicles': len(vehicles),
        }
        
        return annotated_frame, results_data
    
    def _get_lane(self, x, y):
        """Determine which lane a point (x, y) belongs to."""
        if not self.lane_zones:
            return None
        for lane_name, polygon in self.lane_zones.items():
            if cv2.pointPolygonTest(polygon, (int(x), int(y)), False) >= 0:
                return lane_name
        return 'unknown'
    
    def _draw_annotations(self, frame, detections, vehicles, 
                          lane_densities, signal_timings, emergency_status):
        """Draw bounding boxes, lane zones, density info, and signal timers on frame."""
        annotated = detections.plot()
        h, w = annotated.shape[:2]
        
        # Draw lane zones
        if self.lane_zones:
            for lane_name, polygon in self.lane_zones.items():
                cv2.polylines(annotated, [polygon], True, (255, 255, 0), 2)
                # Lane label
                cx = int(np.mean(polygon[:, 0]))
                cy = int(np.min(polygon[:, 1])) - 10
                cv2.putText(annotated, lane_name, (cx - 30, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Info panel (top-right)
        panel_x = w - 300
        panel_y = 10
        
        # Background for info panel
        overlay = annotated.copy()
        cv2.rectangle(overlay, (panel_x - 10, panel_y - 5),
                      (w - 5, panel_y + 220), (0, 0, 0), -1)
        annotated = cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0)
        
        # FPS
        cv2.putText(annotated, f"FPS: {self.fps:.1f}", (panel_x, panel_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Total vehicles
        total = sum(1 for v in vehicles if v['class'] not in EMERGENCY_CLASSES)
        cv2.putText(annotated, f"Vehicles: {total}", (panel_x, panel_y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Lane densities
        y_offset = panel_y + 75
        for lane, density in lane_densities.items():
            color = (0, 255, 0) if density < 20 else (0, 255, 255) if density < 40 else (0, 0, 255)
            cv2.putText(annotated, f"{lane}: D={density:.0f}", (panel_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_offset += 22
        
        # Signal timings
        y_offset += 10
        cv2.putText(annotated, "Signal Timings:", (panel_x, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        y_offset += 22
        for lane, timing in signal_timings.items():
            cv2.putText(annotated, f"  {lane}: {timing}s", (panel_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 20
        
        # Emergency banner
        if emergency_status['mode'] == 'EMERGENCY':
            # Flashing red banner
            if self.frame_count % 20 < 10:  # Blink effect
                cv2.rectangle(annotated, (0, 0), (w, 50), (0, 0, 200), -1)
                cv2.putText(annotated, 
                           f"EMERGENCY VEHICLE DETECTED - GREEN CORRIDOR: {emergency_status.get('green_lane', '?')}",
                           (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return annotated


def run_video(source, model_path, confidence, use_lanes, output_path=None):
    """
    Run detection on video/webcam with live display.
    """
    # Set up lane zones
    lanes = LANE_ZONES if use_lanes else None
    
    # Initialize detector
    detector = TrafficDetector(
        model_path=model_path,
        confidence=confidence,
        lane_zones=lanes
    )
    
    # Open video source
    if source.isdigit():
        source = int(source)
        print(f"📹 Opening webcam (device {source})...")
    else:
        print(f"📹 Opening video: {source}...")
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Could not open video source: {source}")
        return
    
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"   Resolution: {frame_w}x{frame_h}")
    print(f"   FPS: {fps}")
    if total_frames > 0:
        print(f"   Total frames: {total_frames}")
    
    # Video writer (optional)
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
        print(f"   Output: {output_path}")
    
    print("\n🚀 Running detection... Press 'q' to quit")
    print("─" * 50)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, int):
                    continue  # Webcam may have dropped frame
                break  # End of video
            
            # Process frame
            annotated, data = detector.detect_frame(frame)
            
            # Print stats every 30 frames
            if data['frame'] % 30 == 0:
                print(f"  Frame {data['frame']:5d} | "
                      f"FPS: {data['fps']:5.1f} | "
                      f"Vehicles: {data['total_vehicles']:3d} | "
                      f"Density: {data['total_density']:6.1f} | "
                      f"Emergency: {data['emergency']['mode']}")
            
            # Display (skip gracefully if no GUI available)
            try:
                cv2.imshow("Traffic Detection System", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⏹️ Stopped by user")
                    break
                elif key == ord('p'):
                    print("⏸️ Paused. Press any key to continue...")
                    cv2.waitKey(0)
            except cv2.error:
                pass  # No display available — running headlessly
            
            # Write output
            if writer:
                writer.write(annotated)
    
    finally:
        cap.release()
        if writer:
            writer.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    
    print(f"\n✅ Processed {detector.frame_count} frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Detection System")
    parser.add_argument("--source", default="0",
                       help="Video file path, image folder, or webcam device number")
    parser.add_argument("--model", default="models/best.pt",
                       help="Path to YOLOv8 model weights")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="Detection confidence threshold (0-1)")
    parser.add_argument("--lanes", action="store_true",
                       help="Enable per-lane counting")
    parser.add_argument("--output", default=None,
                       help="Save annotated video to this path")
    
    args = parser.parse_args()
    run_video(args.source, args.model, args.conf, args.lanes, args.output)
