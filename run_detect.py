"""
Run full traffic detection on a video file.
Saves annotated output video with vehicle counts, density, and signal timings overlaid.
Output is saved to: runs/detect/predict/
"""
import sys
import cv2
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from detect import TrafficDetector

# ── CONFIG ────────────────────────────────────────────────────
VIDEO_PATH = r"C:\Users\magzt\OneDrive\Desktop\Traffic Congestion Model\Traffic_Management\data\video1.mp4"
MODEL_PATH = "yolov8s.pt"
CONFIDENCE = 0.25
OUTPUT_DIR = Path("runs/detect/predict")
# ─────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = str(OUTPUT_DIR / "output.mp4")

detector = TrafficDetector(model_path=MODEL_PATH, confidence=CONFIDENCE)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Could not open video: {VIDEO_PATH}")
    sys.exit(1)

w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps   = cap.get(cv2.CAP_PROP_FPS) or 30
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

print(f"📹 Processing : {VIDEO_PATH}")
print(f"   Resolution : {w}x{h}  |  FPS: {fps:.0f}  |  Frames: {total}")
print(f"   Output     : {output_path}")
print("   Press Ctrl+C to stop early...\n")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated, data = detector.detect_frame(frame)
        writer.write(annotated)

        if data["frame"] % 30 == 0:
            pct = data["frame"] / total * 100 if total > 0 else 0
            print(f"  Frame {data['frame']:5d}/{total} ({pct:5.1f}%)  |  "
                  f"Vehicles: {data['total_vehicles']:3d}  |  "
                  f"Density: {data['total_density']:6.1f}  |  "
                  f"Emergency: {data['emergency']['mode']}")

        # Show live window if display is available
        try:
            cv2.imshow("Traffic Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n⏹  Stopped by user")
                break
        except cv2.error:
            pass  # No display (headless) — continue saving to file

except KeyboardInterrupt:
    print("\n⏹  Interrupted")

finally:
    cap.release()
    writer.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

print(f"\n✅ Done! Annotated video saved to: {output_path}")
print(f"   Total frames processed: {detector.frame_count}")
