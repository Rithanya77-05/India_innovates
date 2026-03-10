"""
RAKSHA GRID v6.0 — Judge-Ready Accurate Accident Detection
===========================================================

REAL accident = ALL 3 conditions met simultaneously:
  1. OVERLAP     — bounding boxes physically merge (IoU > threshold)
  2. HIGH SPEED  — at least one vehicle was moving fast before impact
  3. SUDDEN STOP — at least one vehicle decelerates sharply after overlap

This eliminates false positives from:
  - Vehicles stopped at traffic lights
  - Vehicles passing close to each other
  - Narrow roads where boxes are always near each other
  - Slow parking / reversing

Run:
  python main.py --source test.mp4 --no-call
  python main.py --source test.mp4 --road highway --no-call
  python main.py --source test.mp4 --show-debug --no-call
"""

import os, json, time, csv, argparse, threading
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# pylint: disable=no-member, invalid-name

from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ═══════════════════════════════════════════════════════════
#  ARGS
# ═══════════════════════════════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument("--source",      default="test.mp4")
parser.add_argument("--model",       default="yolov8s.pt")
parser.add_argument("--road",        default="city",
                    choices=["highway","city","narrow"])
parser.add_argument("--no-save",     action="store_true")
parser.add_argument("--no-call",     action="store_true")
parser.add_argument("--speed-limit", default=60, type=int)
parser.add_argument("--show-debug",  action="store_true")
args = parser.parse_args()


# ═══════════════════════════════════════════════════════════
#  ROAD PROFILES — stricter than before
# ═══════════════════════════════════════════════════════════
ROAD_PROFILES = {
    #          iou    min_spd  decel   confirm  cooldown
    "highway": dict(
        iou_threshold   = 0.25,   # boxes must overlap 25%
        min_speed_kmh   = 20.0,   # must be going at least 20 km/h
        decel_threshold = 0.4,    # speed must drop to 40% of pre-crash
        accident_frames = 4,      # must persist 4 frames
        cooldown        = 180,
    ),
    "city": dict(
        iou_threshold   = 0.30,
        min_speed_kmh   = 12.0,
        decel_threshold = 0.35,
        accident_frames = 5,
        cooldown        = 150,
    ),
    "narrow": dict(
        iou_threshold   = 0.45,   # higher because boxes are naturally close
        min_speed_kmh   = 8.0,
        decel_threshold = 0.30,
        accident_frames = 6,
        cooldown        = 150,
    ),
}
P = ROAD_PROFILES[args.road]


# ═══════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════
VEHICLE_CLASSES  = [1, 2, 3, 5, 6, 7]
PERSON_CLASS     = [0]
ALL_CLASSES      = VEHICLE_CLASSES + PERSON_CLASS
CONF_THRESHOLD   = 0.45

CONFIG_FILE      = "config.json"
INCIDENTS_DIR    = Path("incidents")
LOG_FILE         = "incidents_log.csv"
INCIDENTS_DIR.mkdir(exist_ok=True)

TRAIL_LENGTH        = 50
PREDICT_STEPS       = 20
SPEED_SMOOTH        = 8
SPEED_HISTORY       = 20    # frames of speed history per vehicle

# Parked detection
PARKED_PX_THRESHOLD = 1.2
PARKED_FRAMES       = 20
PARKED_HYSTERESIS   = 8

# Auto-scale
VEHICLE_REAL_LENGTHS = {
    "car":4.5, "truck":8.0, "bus":12.0, "motorcycle":2.2, "bicycle":1.8
}
SCALE_LEARN_SAMPLES  = 15
SCALE_CONFIDENCE_MIN = 5

FLOW_LEARN   = 120
PRE_CLIP_SEC = 6
POST_CLIP_SEC= 6

# Alert thresholds
ALERT_SMS_MIN_SCORE  = 55
ALERT_CALL_MIN_SCORE = 80

# Colors BGR
C_NORMAL   = (0, 220, 0)
C_WARN     = (0, 165, 255)
C_DANGER   = (0, 0, 255)
C_PERSON   = (255, 100, 0)
C_WRONGWAY = (0, 0, 200)
C_OVERSPEED= (0, 200, 255)
C_TRAIL    = (255, 220, 0)
C_PREDICT  = (180, 0, 255)
C_TEXT     = (255, 255, 255)
C_HUD      = (15, 15, 25)
C_PARKED   = (100, 100, 100)


# ═══════════════════════════════════════════════════════════
#  LOAD CONFIG
# ═══════════════════════════════════════════════════════════
DEFAULT_CFG = {
    "twilio_sid":"","twilio_auth":"","twilio_from":"","alert_to":"",
    "camera_lat":10.6578,"camera_lng":77.0083,
    "camera_location":"Unknown Location","camera_id":"CAM-001"
}
cfg = DEFAULT_CFG.copy()
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE) as f: cfg.update(json.load(f))


# ═══════════════════════════════════════════════════════════
#  AUTO SCALE ESTIMATOR
# ═══════════════════════════════════════════════════════════
class AutoScaleEstimator:
    def __init__(self):
        self.samples=[]; self.mpp=None; self.locked=False; self.n=0

    def update(self, label, bw, bh):
        if self.locked: return
        real=VEHICLE_REAL_LENGTHS.get(label)
        if real is None: return
        px=max(bw,bh)
        if px<10: return
        self.samples.append(real/px); self.n+=1
        if self.n>=SCALE_LEARN_SAMPLES:
            self.mpp=float(np.median(self.samples)); self.locked=True
            print(f"[SCALE] Locked: {self.mpp:.4f} m/px from {self.n} samples")

    @property
    def ready(self): return self.n>=SCALE_CONFIDENCE_MIN

    @property
    def current_mpp(self):
        if not self.ready: return None
        return self.mpp if self.locked else float(np.median(self.samples))

    def px_to_kmh(self, px_per_frame, fps):
        m=self.current_mpp
        return px_per_frame*m*fps*3.6 if m else None

    def status(self):
        if self.locked:   return f"CAL:LOCKED({self.n})", C_NORMAL
        if self.ready:    return f"CAL:{self.n}/{SCALE_LEARN_SAMPLES}", C_WARN
        return                   f"WARMING:{self.n}/{SCALE_CONFIDENCE_MIN}", (100,100,100)


# ═══════════════════════════════════════════════════════════
#  PARKED TRACKER
# ═══════════════════════════════════════════════════════════
class ParkedTracker:
    def __init__(self):
        self.still=defaultdict(int); self.moving=defaultdict(int)
        self.parked=defaultdict(bool)

    def update(self, tid, px_spd):
        if px_spd<PARKED_PX_THRESHOLD:
            self.still[tid]+=1; self.moving[tid]=0
        else:
            self.moving[tid]+=1; self.still[tid]=0
        if self.still[tid]>=PARKED_FRAMES:   self.parked[tid]=True
        if self.moving[tid]>=PARKED_HYSTERESIS: self.parked[tid]=False
        return self.parked[tid]


# ═══════════════════════════════════════════════════════════
#  VEHICLE SPEED HISTORY
#  Stores per-vehicle speed history to detect sudden deceleration
# ═══════════════════════════════════════════════════════════
class SpeedHistory:
    """
    Tracks speed over time per vehicle.
    Used to detect:
      - Pre-crash speed (was vehicle moving fast?)
      - Post-crash deceleration (did it suddenly slow down?)
    """
    def __init__(self):
        self.history = defaultdict(lambda: deque(maxlen=SPEED_HISTORY))

    def update(self, tid, kmh):
        self.history[tid].append(kmh)

    def pre_crash_speed(self, tid, lookback=10):
        """Max speed in the last N frames before now."""
        h=list(self.history[tid])
        if not h: return 0.0
        window=h[:-3] if len(h)>3 else h   # exclude very recent frames
        return float(max(window)) if window else 0.0

    def sudden_decel(self, tid):
        """
        Returns True if speed dropped sharply.
        Compares average of first half vs second half of history.
        """
        h=list(self.history[tid])
        if len(h)<8: return False
        mid=len(h)//2
        before=float(np.mean(h[:mid]))
        after =float(np.mean(h[mid:]))
        if before<2.0: return False   # was barely moving — not a crash
        ratio=after/(before+1e-6)
        return ratio<P["decel_threshold"]   # dropped to <35% of earlier speed


# ═══════════════════════════════════════════════════════════
#  ACCIDENT VALIDATOR
#  The core judge — only confirms accident if ALL 3 conditions met
# ═══════════════════════════════════════════════════════════
class AccidentValidator:
    """
    3-condition gate for real accident confirmation:
      C1: IoU overlap above threshold (physical contact)
      C2: At least one vehicle had high speed before impact
      C3: At least one vehicle shows sudden deceleration after overlap

    Tracks candidate pairs and only fires when all 3 are true
    for enough consecutive frames.
    """
    def __init__(self):
        self.candidates = {}   # (id_a, id_b) → frame count

    def check(self, id_a, id_b, iou_val, spd_a, spd_b,
              decel_a, decel_b, pre_spd_a, pre_spd_b):
        key = (min(id_a,id_b), max(id_a,id_b))

        # C1: Physical overlap
        c1 = iou_val > P["iou_threshold"]

        # C2: At least one was going fast
        min_spd = P["min_speed_kmh"]
        c2 = pre_spd_a > min_spd or pre_spd_b > min_spd

        # C3: At least one suddenly decelerated
        c3 = decel_a or decel_b

        if c1 and c2 and c3:
            self.candidates[key] = self.candidates.get(key, 0) + 1
        else:
            # Partial match — only keep candidate if overlap exists
            if not c1:
                self.candidates.pop(key, None)
            # If overlap exists but C2/C3 not yet met, give grace period
            elif key in self.candidates:
                pass   # keep counting frames

        confirmed = self.candidates.get(key, 0) >= P["accident_frames"]

        if args.show_debug:
            status = f"C1:{int(c1)} C2:{int(c2)} C3:{int(c3)} f:{self.candidates.get(key,0)}"
            return confirmed, status
        return confirmed, None

    def reset(self, id_a, id_b):
        key=(min(id_a,id_b),max(id_a,id_b))
        self.candidates.pop(key,None)


# ═══════════════════════════════════════════════════════════
#  SEVERITY ENGINE
# ═══════════════════════════════════════════════════════════
class SeverityEngine:
    def score(self, max_kmh, num_vehicles, has_ped, stopped_after):
        if max_kmh>=80:   s=40
        elif max_kmh>=40: s=25
        else:             s=10
        s+=min(num_vehicles*8,20)
        s+=25 if has_ped else 0
        s+=15 if stopped_after else 0
        return min(s,100)

    def label(self, score):
        if score>=80: return "CRITICAL",(0,  0,160)
        if score>=55: return "SEVERE",  (0,  0,220)
        if score>=30: return "MODERATE",(0,100,255)
        return               "MINOR",   (0,165,255)

    def needs_call(self,score): return score>=ALERT_CALL_MIN_SCORE
    def needs_sms(self,score):  return score>=ALERT_SMS_MIN_SCORE
    def is_major(self,score):   return score>=ALERT_SMS_MIN_SCORE


# ═══════════════════════════════════════════════════════════
#  EMERGENCY ALERT
# ═══════════════════════════════════════════════════════════
class EmergencyAlert:
    def __init__(self):
        self.twilio_ok=(
            not args.no_call and
            bool(cfg["twilio_sid"]) and bool(cfg["alert_to"])
        )
        if self.twilio_ok:
            try:
                from twilio.rest import Client
                self.client=Client(cfg["twilio_sid"],cfg["twilio_auth"])
                print("[ALERT] Twilio: READY")
            except ImportError:
                print("[ALERT] pip install twilio"); self.twilio_ok=False
        else:
            print("[ALERT] Mode: TEST")

    def dispatch(self,sev_label,sev_score,n_veh,max_kmh,has_ped,iid,sev_eng):
        do_call=sev_eng.needs_call(sev_score) and self.twilio_ok
        do_sms =sev_eng.needs_sms(sev_score)
        if not do_sms and not do_call:
            print(f"[INFO] Minor incident {iid} — warning only, no alert")
            return
        threading.Thread(
            target=self._send,
            args=(sev_label,sev_score,n_veh,max_kmh,has_ped,iid,do_call,do_sms),
            daemon=True
        ).start()

    def _send(self,sev_label,sev_score,n_veh,max_kmh,has_ped,iid,do_call,do_sms):
        lat=cfg["camera_lat"]; lng=cfg["camera_lng"]
        loc=cfg["camera_location"]
        maps=f"https://maps.google.com/?q={lat},{lng}"
        ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        voice=(
            f"Hello. This is RAKSHA GRID. "
            f"A {sev_label} accident has been detected at {loc}. "
            f"Coordinates: Latitude {lat}, Longitude {lng}. "
            f"Time: {ts}. Vehicles: {n_veh}. "
            f"{'Pedestrian involved. ' if has_ped else ''}"
            f"Speed: {max_kmh:.0f} km/h. Score: {sev_score} out of 100. "
            f"Please dispatch ambulance. ID: {iid}."
        )
        sms=(
            f"RAKSHA GRID ACCIDENT\n"
            f"ID:{iid}\n{sev_label}({sev_score}/100)\n"
            f"{loc}\n{maps}\n"
            f"V:{n_veh} spd:{max_kmh:.0f}km/h\n"
            f"Ped:{'YES' if has_ped else 'No'}\n{ts}"
        )
        if self.twilio_ok:
            try:
                if do_call:
                    c=self.client.calls.create(
                        twiml=f"<Response><Say voice='alice' language='en-IN'>{voice}</Say></Response>",
                        to=cfg["alert_to"],from_=cfg["twilio_from"])
                    print(f"[CALL] {c.sid}")
                if do_sms:
                    m=self.client.messages.create(
                        body=sms,to=cfg["alert_to"],from_=cfg["twilio_from"])
                    print(f"[SMS] {m.sid}")
            except Exception as e: print(f"[ERROR] {e}")
        else:
            action="CALL+SMS" if do_call else "SMS"
            print(f"\n{'═'*50}")
            print(f"  [TEST {action}] {iid}")
            print(f"  {sev_label} ({sev_score}/100) | {max_kmh:.0f}km/h")
            print(f"  {loc} | {maps}")
            print(f"{'═'*50}\n")


# ═══════════════════════════════════════════════════════════
#  INCIDENT REPORTER
# ═══════════════════════════════════════════════════════════
class IncidentReporter:
    def __init__(self):
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE,"w",newline="") as f:
                csv.writer(f).writerow([
                    "id","timestamp","type","severity","score",
                    "vehicles","max_kmh","pedestrian",
                    "lat","lng","location","alert","clip"
                ])

    def new_id(self): return f"RG-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    def save(self,iid,itype,sl,sc,vids,max_kmh,has_ped,alert,clip=None):
        ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE,"a",newline="") as f:
            csv.writer(f).writerow([
                iid,ts,itype,sl,sc,str(vids),f"{max_kmh:.1f}",
                has_ped,cfg["camera_lat"],cfg["camera_lng"],
                cfg["camera_location"],alert,clip or ""])
        rp=str(INCIDENTS_DIR/f"{iid}.json")
        with open(rp,"w") as f:
            json.dump({
                "id":iid,"ts":ts,"cam":cfg["camera_id"],
                "location":cfg["camera_location"],
                "coords":{"lat":cfg["camera_lat"],"lng":cfg["camera_lng"]},
                "maps":f"https://maps.google.com/?q={cfg['camera_lat']},{cfg['camera_lng']}",
                "type":itype,"severity":sl,"score":sc,
                "vehicles":vids,"speed_kmh":round(max_kmh,1),
                "pedestrian":has_ped,"alert":alert,"clip":clip or ""
            },f,indent=2)
        print(f"[REPORT] {sl}({sc}) {iid}")


# ═══════════════════════════════════════════════════════════
#  CLIP SAVER
# ═══════════════════════════════════════════════════════════
class ClipSaver:
    def __init__(self,fps,w,h):
        self.fps=fps; self.w=w; self.h=h
        self.buf=deque(maxlen=int(fps*PRE_CLIP_SEC))
        self.recording=False; self.post=0
        self.writer=None; self.path=None

    def push(self,frame):
        if not self.recording: self.buf.append(frame.copy())
        else:
            if self.writer: self.writer.write(frame)
            self.post-=1
            if self.post<=0: self._stop()

    def trigger(self,iid):
        if self.recording: return self.path
        self.path=str(INCIDENTS_DIR/f"{iid}.avi")
        self.writer=cv2.VideoWriter(
            self.path,cv2.VideoWriter_fourcc(*"XVID"),
            self.fps,(self.w,self.h))
        for f in self.buf: self.writer.write(f)
        self.recording=True; self.post=int(self.fps*POST_CLIP_SEC)
        return self.path

    def _stop(self):
        if self.writer: self.writer.release(); self.writer=None
        self.recording=False; print(f"[CLIP] {self.path}")


# ═══════════════════════════════════════════════════════════
#  KALMAN TRACKER
# ═══════════════════════════════════════════════════════════
class KalmanTracker:
    def __init__(self,cx,cy):
        self.kf=cv2.KalmanFilter(4,2)
        self.kf.measurementMatrix   =np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.kf.transitionMatrix    =np.array([[1,0,1,0],[0,1,0,1],
                                                [0,0,1,0],[0,0,0,1]],np.float32)
        self.kf.processNoiseCov     =np.eye(4,dtype=np.float32)*0.03
        self.kf.measurementNoiseCov =np.eye(2,dtype=np.float32)*0.5
        self.kf.errorCovPost        =np.eye(4,dtype=np.float32)
        self.kf.statePost           =np.array([[float(cx)],[float(cy)],
                                                [0.],[0.]],np.float32)

    def update(self,cx,cy):
        self.kf.predict()
        self.kf.correct(np.array([[np.float32(cx)],[np.float32(cy)]]))
        s=self.kf.statePost
        return int(s[0,0]),int(s[1,0])


# ═══════════════════════════════════════════════════════════
#  WRONG-WAY DETECTOR
# ═══════════════════════════════════════════════════════════
class WrongWayDetector:
    def __init__(self):
        self.vecs=[]; self.dom=None; self.learned=False; self.fc=0

    def update(self,trails):
        if self.learned: return
        self.fc+=1
        for tr in trails.values():
            if len(tr)>=3:
                v=np.array(tr[-1])-np.array(tr[-3],dtype=float)
                n=np.linalg.norm(v)
                if n>2: self.vecs.append(v/n)
        if self.fc>=FLOW_LEARN and self.vecs:
            avg=np.mean(self.vecs,axis=0); n=np.linalg.norm(avg)
            if n>0:
                self.dom=avg/n; self.learned=True
                print(f"[FLOW] Direction learned: "
                      f"{np.degrees(np.arctan2(self.dom[1],self.dom[0])):.1f}°")

    def is_wrong_way(self,trail):
        if not self.learned or len(trail)<4: return False
        v=np.array(trail[-1])-np.array(trail[-4],dtype=float)
        n=np.linalg.norm(v)
        if n<3: return False
        return float(np.dot(v/n,self.dom))<-0.6


# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════
def compute_iou(b1,b2):
    x1=max(b1[0],b2[0]); y1=max(b1[1],b2[1])
    x2=min(b1[2],b2[2]); y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    u=(b1[2]-b1[0])*(b1[3]-b1[1])+(b2[2]-b2[0])*(b2[3]-b2[1])-inter
    return inter/u if u>0 else 0

def predict_traj(trail,steps):
    if len(trail)<2: return []
    n=min(len(trail),6); pts=np.array(trail[-n:],dtype=float)
    w=np.arange(1,n,dtype=float); d=np.diff(pts,axis=0)
    vel=(d*(w/w.sum())[:,None]).sum(axis=0)
    out=[]; x,y=pts[-1]
    for _ in range(steps):
        x+=vel[0]; y+=vel[1]; out.append((int(x),int(y)))
    return out

def draw_label(frame,text,pos,color):
    x,y=pos
    (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.45,1)
    cv2.rectangle(frame,(x-2,y-th-4),(x+tw+2,y+2),(0,0,0),-1)
    cv2.putText(frame,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)


# ═══════════════════════════════════════════════════════════
#  INIT
# ═══════════════════════════════════════════════════════════
model   = YOLO(args.model)
src     = int(args.source) if args.source.isdigit() else args.source
cap     = cv2.VideoCapture(src)
vfps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
fw      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

scale    = AutoScaleEstimator()
parked   = ParkedTracker()
spd_hist = SpeedHistory()
acc_val  = AccidentValidator()
wwd      = WrongWayDetector()
sev_eng  = SeverityEngine()
alerter  = EmergencyAlert()
rep      = IncidentReporter()
clips    = ClipSaver(vfps,fw,fh) if not args.no_save else None

trails      = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
px_spd_hist = defaultdict(lambda: deque(maxlen=SPEED_SMOOTH))
kalmans     = {}

# Single cooldown state
cooldown    = 0
acc_done    = False
wway_state  = {"f":0,"cd":0,"done":False}
ovspd_state = {"f":0,"cd":0,"done":False}
ped_state   = {"f":0,"cd":0,"done":False}

fps_t=time.time(); fps_v=0.0; fc=0

print(f"""
╔══════════════════════════════════════════════╗
║        RAKSHA GRID v6.0                      ║
╠══════════════════════════════════════════════╣
║  Road    : {args.road.upper():<33}║
║  Source  : {str(args.source)[:33]:<33}║
║  FPS     : {vfps:<33.1f}║
╠══════════════════════════════════════════════╣
║  ACCIDENT = overlap + high speed + decel     ║
║  All 3 conditions must be true               ║
╠══════════════════════════════════════════════╣
║  MINOR/MODERATE → WARNING on screen only     ║
║  SEVERE         → SMS alert                  ║
║  CRITICAL       → Voice call + SMS           ║
╚══════════════════════════════════════════════╝
  ESC = quit  |  --show-debug for condition values
""")

SPEED_SMOOTH = 8


# ═══════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════
while True:
    ret,frame=cap.read()
    if not ret: print("[INFO] Stream ended."); break
    fc+=1

    results=model.track(
        frame,persist=True,tracker="bytetrack.yaml",
        classes=ALL_CLASSES,conf=CONF_THRESHOLD,verbose=False
    )

    vehicles=[]; pedestrians=[]

    for r in results:
        for box in r.boxes:
            if box.id is None: continue
            tid=int(box.id[0]); cls=int(box.cls[0])
            label=model.names[cls]
            x1,y1,x2,y2=map(int,box.xyxy[0])
            bw,bh=(x2-x1),(y2-y1)
            rx,ry=(x1+x2)//2,(y1+y2)//2
            is_person=(cls in PERSON_CLASS)

            if not is_person: scale.update(label,bw,bh)

            if tid not in kalmans: kalmans[tid]=KalmanTracker(rx,ry)
            cx,cy=kalmans[tid].update(rx,ry)
            trails[tid].append((cx,cy))

            # Pixel speed
            if len(trails[tid])>=2:
                prev=trails[tid][-2]
                px_spd_hist[tid].append(
                    float(np.hypot(cx-prev[0],cy-prev[1])))
            avg_px=float(np.mean(px_spd_hist[tid])) if px_spd_hist[tid] else 0.0

            is_stat=parked.update(tid,avg_px) if not is_person else False

            # km/h
            kmh_val=scale.px_to_kmh(avg_px,vfps)
            kmh=kmh_val if (kmh_val and not is_stat) else 0.0

            # Feed speed history (for decel detection)
            if not is_person:
                spd_hist.update(tid, kmh if scale.ready else avg_px)

            # Speed display
            if is_person:        spd_str=""
            elif is_stat:        spd_str="0 km/h"
            elif scale.ready:    spd_str=f"{kmh:.0f} km/h"
            else:                spd_str=f"{avg_px:.1f}px/f"

            preds=predict_traj(list(trails[tid]),PREDICT_STEPS)
            d=dict(
                id=tid,box=(x1,y1,x2,y2),center=(cx,cy),
                px_spd=avg_px,kmh=kmh,spd_str=spd_str,
                preds=preds,label=label,
                is_person=is_person,is_parked=is_stat
            )
            (pedestrians if is_person else vehicles).append(d)

    wwd.update(trails)

    # ══════════════════════════════════════════
    #  ACCIDENT DETECTION — 3-condition gate
    # ══════════════════════════════════════════
    accident_confirmed = False
    danger_ids         = {}
    crash_ids          = []
    crash_kmhs         = []
    ovspd_ids          = set()
    wway_ids           = set()
    ped_ids            = set(p["id"] for p in pedestrians)

    for i,da in enumerate(vehicles):
        for j in range(i+1,len(vehicles)):
            db=vehicles[j]

            # Skip parked pairs
            if da["is_parked"] and db["is_parked"]: continue

            tid_a,tid_b=da["id"],db["id"]
            iou_val=compute_iou(da["box"],db["box"])

            # Pre-crash speed
            pre_a=spd_hist.pre_crash_speed(tid_a)
            pre_b=spd_hist.pre_crash_speed(tid_b)

            # Deceleration check
            decel_a=spd_hist.sudden_decel(tid_a)
            decel_b=spd_hist.sudden_decel(tid_b)

            confirmed,dbg=acc_val.check(
                tid_a,tid_b,iou_val,
                da["kmh"],db["kmh"],
                decel_a,decel_b,pre_a,pre_b
            )

            if confirmed:
                accident_confirmed=True
                for d in (da,db):
                    danger_ids[d["id"]]="collision"
                    crash_ids.append(d["id"])
                    crash_kmhs.append(d["kmh"] or pre_a or pre_b)

            elif iou_val>P["iou_threshold"]*0.6:
                # Partial — overlap exists but not all conditions met yet
                for d in (da,db):
                    danger_ids.setdefault(d["id"],"warn")

            if args.show_debug and dbg:
                mx=(da["center"][0]+db["center"][0])//2
                my=(da["center"][1]+db["center"][1])//2
                draw_label(frame,
                    f"IoU:{iou_val:.2f} {dbg}",
                    (mx,my),(180,180,180))

    # Wrong way
    for d in vehicles:
        if not d["is_parked"] and wwd.is_wrong_way(list(trails[d["id"]])):
            wway_ids.add(d["id"]); danger_ids[d["id"]]="wrong_way"

    # Overspeed
    if scale.ready:
        for d in vehicles:
            if not d["is_parked"] and d["kmh"]>args.speed_limit:
                ovspd_ids.add(d["id"])

    # ── ACCIDENT STATE MACHINE ───────────────
    confirmed_incidents = set()
    sev_info = {}   # key → (score, label, max_kmh)

    # Collision
    if cooldown>0:
        cooldown-=1
    else:
        if not accident_confirmed:
            acc_done=False
        if accident_confirmed:
            confirmed_incidents.add("collision")
            max_kmh=max(crash_kmhs) if crash_kmhs else 0.0
            n_veh=len(set(crash_ids)) or 1
            has_ped=bool(pedestrians)
            stopped=any(parked.still.get(t,0)>PARKED_FRAMES for t in crash_ids)
            sc=sev_eng.score(max_kmh,n_veh,has_ped,stopped)
            sl,_=sev_eng.label(sc)
            sev_info["collision"]=(sc,sl,max_kmh)

            if not acc_done:
                iid=rep.new_id()
                clip=clips.trigger(iid) if (clips and sev_eng.is_major(sc)) else None
                alert=("call+sms" if sev_eng.needs_call(sc)
                       else "sms" if sev_eng.needs_sms(sc) else "none")
                rep.save(iid,"VEHICLE COLLISION",sl,sc,
                         list(set(crash_ids)),max_kmh,has_ped,alert,clip)
                alerter.dispatch(sl,sc,n_veh,max_kmh,has_ped,iid,sev_eng)
                acc_done=True; cooldown=P["cooldown"]

    # Wrong way / overspeed / pedestrian — simple state machines
    for key,firing,ids in [
        ("wrong_way", bool(wway_ids),  wway_ids),
        ("overspeed", bool(ovspd_ids), ovspd_ids),
        ("pedestrian",bool(ped_ids),   ped_ids),
    ]:
        s=wway_state if key=="wrong_way" else \
          ovspd_state if key=="overspeed" else ped_state
        if s["cd"]>0: s["cd"]-=1
        else:
            s["f"]=s["f"]+1 if firing else 0
            if not firing: s["done"]=False
            if s["f"]>=5:
                confirmed_incidents.add(key)
                if not s["done"]:
                    iid=rep.new_id()
                    max_k=max((v["kmh"] for v in vehicles if v["id"] in ids),default=0.0)
                    rep.save(iid,key.upper().replace("_"," "),"INFO",0,
                             list(ids),max_k,key=="pedestrian","none")
                    s["done"]=True; s["cd"]=120

    if clips: clips.push(frame)

    # ══════════════════════════════════════════
    #  DRAW
    # ══════════════════════════════════════════
    for d in vehicles+pedestrians:
        tid=d["id"]; x1,y1,x2,y2=d["box"]
        it=danger_ids.get(tid)
        col=(C_PARKED    if d["is_parked"] else
             C_DANGER    if it=="collision" else
             C_WRONGWAY  if it=="wrong_way" else
             C_WARN      if it=="warn"      else
             C_OVERSPEED if tid in ovspd_ids else
             C_PERSON    if d["is_person"]  else C_NORMAL)

        cv2.rectangle(frame,(x1,y1),(x2,y2),col,
                      3 if it in ("collision","wrong_way") else 2)

        tags=[]
        if d["is_parked"] and not d["is_person"]: tags.append("PARKED")
        if tid in wway_ids:  tags.append("WRONG WAY!")
        if tid in ovspd_ids: tags.append("FAST!")
        if tid in ped_ids:   tags.append("PED")
        tag_s=("  "+" ".join(tags)) if tags else ""

        draw_label(frame,
                   f"ID:{tid} {d['label']} {d['spd_str']}{tag_s}",
                   (x1,y1-8),col)

        if not d["is_parked"]:
            tp=list(trails[tid])
            for k in range(1,len(tp)):
                cv2.line(frame,tp[k-1],tp[k],C_TRAIL,max(1,k*2//len(tp)))

        if d["preds"] and not d["is_person"] and not d["is_parked"]:
            for k in range(1,len(d["preds"])):
                cv2.line(frame,d["preds"][k-1],d["preds"][k],C_PREDICT,1)
            cv2.circle(frame,d["preds"][-1],4,C_PREDICT,-1)

    # ── BANNERS ──────────────────────────────
    by=0; BH=55

    if "collision" in confirmed_incidents:
        sc,sl,max_k=sev_info.get("collision",(0,"MINOR",0))
        is_major=sev_eng.is_major(sc)
        _,scol=sev_eng.label(sc)
        title="ACCIDENT DETECTED" if is_major else "MINOR COLLISION"
        bcol=(0,0,150) if is_major else (0,100,200)

        cv2.rectangle(frame,(0,by),(fw,by+BH),bcol,-1)
        cv2.putText(frame,title,(18,by+38),
                    cv2.FONT_HERSHEY_SIMPLEX,1.2,C_TEXT,3)
        cv2.rectangle(frame,(fw-175,by),(fw,by+BH),scol,-1)
        cv2.putText(frame,sl,(fw-170,by+28),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,C_TEXT,2)
        cv2.putText(frame,f"{sc}/100 | {max_k:.0f}km/h",(fw-170,by+50),
                    cv2.FONT_HERSHEY_SIMPLEX,0.48,C_TEXT,1)
        alert_tag=("CALL+SMS" if sev_eng.needs_call(sc)
                   else "SMS" if sev_eng.needs_sms(sc) else "WARNING ONLY")
        draw_label(frame,alert_tag,(20,by+BH-8),
                   (255,255,100) if not is_major else C_TEXT)
        by+=BH

    for key,color,title in [
        ("wrong_way", (0,0,200),   "WRONG-WAY VEHICLE"),
        ("pedestrian",(160,60,0),  "PEDESTRIAN ON ROAD"),
        ("overspeed", (0,120,200), "OVERSPEED DETECTED"),
    ]:
        if key in confirmed_incidents:
            cv2.rectangle(frame,(0,by),(fw,by+BH),color,-1)
            cv2.putText(frame,title,(18,by+38),
                        cv2.FONT_HERSHEY_SIMPLEX,1.2,C_TEXT,3)
            by+=BH

    # ── HUD ──────────────────────────────────
    if fc%15==0: fps_v=15/(time.time()-fps_t+1e-6); fps_t=time.time()
    h=frame.shape[0]
    cv2.rectangle(frame,(0,h-58),(fw,h),C_HUD,-1)
    status=("ACCIDENT" if "collision" in confirmed_incidents else
            "WRONG WAY" if "wrong_way" in confirmed_incidents else
            "OVERSPEED" if "overspeed" in confirmed_incidents else
            "PEDESTRIAN" if "pedestrian" in confirmed_incidents else "NORMAL")
    scol=C_DANGER if confirmed_incidents else C_NORMAL
    cv2.putText(frame,f"V:{len(vehicles)} P:{len(pedestrians)}",
                (10,h-36),cv2.FONT_HERSHEY_SIMPLEX,0.55,C_TEXT,1)
    cv2.putText(frame,f"STATUS:{status}",
                (10,h-12),cv2.FONT_HERSHEY_SIMPLEX,0.52,scol,2)
    cv2.putText(frame,f"FPS:{fps_v:.1f}",
                (fw-220,h-36),cv2.FONT_HERSHEY_SIMPLEX,0.55,C_TEXT,1)
    ss,sc2=scale.status()
    cv2.putText(frame,ss,(fw-220,h-12),cv2.FONT_HERSHEY_SIMPLEX,0.42,sc2,1)

    cv2.imshow("RAKSHA GRID v6.0",frame)
    if cv2.waitKey(1)==27: break

cap.release()
cv2.destroyAllWindows()
print(f"\n[DONE] Log:{LOG_FILE} | Clips:{INCIDENTS_DIR}/")