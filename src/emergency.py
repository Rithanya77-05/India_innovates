"""
============================================
Emergency Vehicle Detection & Green Corridor
============================================
Detects emergency vehicles (ambulance, fire truck) and
triggers green corridor override.

Logic (Updated):
  1. YOLO detects ambulance/fire_truck in a frame OR GPS trigger received
  2. EmergencyController activates EMERGENCY mode
  3. Signal optimizer overrides: ambulance lane → GREEN, all other lanes → RED
  4. Mode stays active until emergency vehicle not seen for COOLDOWN seconds
  5. Then returns to NORMAL adaptive mode

State Machine:
  NORMAL    → No emergency, adaptive signal timing active
  EMERGENCY → Emergency vehicle detected in specific lane (that lane GREEN, others RED)
  NORMAL    ← Returned after cooldown / ambulance passes
"""

import time


class EmergencyController:
    """
    Emergency vehicle detection and green corridor controller (4 lanes).

    Only the lane where the ambulance is detected turns GREEN.
    All other lanes turn RED immediately.
    After ambulance passes (cooldown expires), system returns to NORMAL.
    """

    LANES = ['lane_1', 'lane_2', 'lane_3', 'lane_4']

    def __init__(self, cooldown: float = 8.0):
        """
        Args:
            cooldown: Seconds to wait after last emergency detection
                      before returning to NORMAL mode.
        """
        self.mode = 'NORMAL'
        self.active = False
        self.emergency_lane = None
        self.emergency_class = None
        self.last_detection_time = 0.0
        self.activation_time = 0.0
        self.cooldown = cooldown
        self.activation_count = 0
        self.history = []           # Log of emergency events
        self.gps_triggered = False  # Was this a GPS-triggered emergency?

    # ------------------------------------------------------------------
    # Main per-frame update
    # ------------------------------------------------------------------

    def check_emergency(self,
                        emergency_detected: bool,
                        lane: str = None,
                        vehicle_class: str = None) -> dict:
        """
        Call every frame with detection results.

        Args:
            emergency_detected: True if ambulance/fire_truck found in frame
            lane:               Which lane the emergency vehicle is in
            vehicle_class:      'Ambulance' or 'fire_truck'

        Returns:
            dict with keys:
                mode            – 'NORMAL' | 'EMERGENCY'
                green_lane      – lane getting GREEN signal (emergency only)
                red_lanes       – list of lanes staying RED (emergency only)
                emergency_class – vehicle type
                time_active     – seconds since activation (emergency only)
                cooldown_remaining – seconds left in cooldown (if cooling down)
                gps_triggered   – True if triggered by GPS
        """
        current_time = time.time()

        if emergency_detected:
            if not self.active:
                # New activation
                self.activation_count += 1
                self.activation_time = current_time
                self._log_event('ACTIVATED', lane, vehicle_class, current_time)
                print(f"\n🚨 EMERGENCY ACTIVATED — {vehicle_class or 'emergency vehicle'} "
                      f"in {lane or 'unknown lane'}")
                print(f"   🟢 {lane} → GREEN   |   All other lanes → RED")

            self.active = True
            self.mode = 'EMERGENCY'
            self.emergency_lane = lane or self.emergency_lane
            self.emergency_class = vehicle_class or self.emergency_class
            self.last_detection_time = current_time

            return self._emergency_payload(current_time)

        # No emergency vehicle in current frame
        if self.active:
            time_since_last = current_time - self.last_detection_time

            if time_since_last > self.cooldown:
                # Cooldown expired → return to normal
                duration = round(current_time - self.activation_time, 1)
                self._log_event('DEACTIVATED', self.emergency_lane,
                                self.emergency_class, current_time,
                                extra={'duration': duration})
                print(f"\n✅ EMERGENCY DEACTIVATED — ambulance passed after {duration}s")
                print("   Returning to NORMAL adaptive mode\n")

                self.active = False
                self.mode = 'NORMAL'
                self.emergency_lane = None
                self.emergency_class = None
                self.gps_triggered = False
                return {'mode': 'NORMAL'}
            else:
                # Still in cooldown → maintain emergency mode
                return {
                    **self._emergency_payload(current_time),
                    'cooldown_remaining': round(self.cooldown - time_since_last, 1),
                }

        return {'mode': 'NORMAL'}

    # ------------------------------------------------------------------
    # GPS-based trigger (no camera needed)
    # ------------------------------------------------------------------

    def trigger_gps(self, ambulance_id: str = 'AMB-001',
                    lane: str = 'lane_1') -> dict:
        """
        Activate emergency mode via GPS signal (no YOLO detection required).

        Args:
            ambulance_id: Identifier of the ambulance unit
            lane:         Which lane to clear (e.g. 'lane_2')

        Returns:
            Current emergency status dict
        """
        current_time = time.time()

        if lane not in self.LANES:
            lane = 'lane_1'  # fallback

        if not self.active:
            self.activation_count += 1
            self.activation_time = current_time
            self._log_event('GPS_ACTIVATED', lane, f'GPS:{ambulance_id}',
                            current_time)
            print(f"\n📡 GPS EMERGENCY TRIGGERED — Ambulance {ambulance_id} "
                  f"approaching {lane}")
            print(f"   🟢 {lane} → GREEN   |   All other lanes → RED")

        self.active = True
        self.mode = 'EMERGENCY'
        self.emergency_lane = lane
        self.emergency_class = f'Ambulance (GPS:{ambulance_id})'
        self.last_detection_time = current_time
        self.gps_triggered = True

        return self._emergency_payload(current_time)

    # ------------------------------------------------------------------
    # Manual deactivation
    # ------------------------------------------------------------------

    def force_deactivate(self) -> dict:
        """Manually deactivate emergency mode (operator override)."""
        current_time = time.time()
        if self.active:
            duration = round(current_time - self.activation_time, 1)
            self._log_event('FORCE_DEACTIVATED', self.emergency_lane,
                            self.emergency_class, current_time,
                            extra={'duration': duration})
            print(f"\n🛑 Emergency MANUALLY deactivated after {duration}s")

        self.active = False
        self.mode = 'NORMAL'
        self.emergency_lane = None
        self.emergency_class = None
        self.gps_triggered = False
        return {'mode': 'NORMAL', 'message': 'Emergency deactivated by operator'}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Get full emergency system statistics."""
        return {
            'current_mode': self.mode,
            'total_activations': self.activation_count,
            'currently_active': self.active,
            'emergency_lane': self.emergency_lane,
            'gps_triggered': self.gps_triggered,
            'history': self.history[-10:],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _emergency_payload(self, current_time: float) -> dict:
        """Build the emergency status dict returned to callers."""
        red_lanes = [l for l in self.LANES if l != self.emergency_lane]
        return {
            'mode': 'EMERGENCY',
            'green_lane': self.emergency_lane,
            'red_lanes': red_lanes,
            'emergency_class': self.emergency_class,
            'time_active': round(current_time - self.activation_time, 1),
            'gps_triggered': self.gps_triggered,
        }

    def _log_event(self, event: str, lane: str, vehicle: str,
                   ts: float, extra: dict = None):
        entry = {
            'time': round(ts, 2),
            'event': event,
            'lane': lane,
            'vehicle': vehicle,
        }
        if extra:
            entry.update(extra)
        self.history.append(entry)
