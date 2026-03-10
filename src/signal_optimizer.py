"""
============================================
Adaptive Signal Timing Optimizer (4 Lanes)
============================================
Dynamically calculates green signal durations
based on real-time traffic density per lane.

Modes:
  NORMAL    - Proportional green time based on density
  EMERGENCY - Ambulance lane → GREEN, all others → RED
"""


class SignalOptimizer:
    """
    Adaptive traffic signal timing controller (supports 4 lanes).

    Normal mode: Proportional allocation based on density.
    Emergency mode: Ambulance lane gets GREEN; all other lanes get RED.
    """

    ALL_LANES = ['lane_1', 'lane_2', 'lane_3', 'lane_4']

    def __init__(self, max_cycle=120, min_green=10, max_green=60,
                 default_green=30):
        """
        Args:
            max_cycle:     Total signal cycle duration in seconds
            min_green:     Minimum green time per lane
            max_green:     Maximum green time per lane
            default_green: Default green when no density data
        """
        self.max_cycle = max_cycle
        self.min_green = min_green
        self.max_green = max_green
        self.default_green = default_green
        self.current_timings = {}
        self.mode = 'NORMAL'
        self._emergency_lane = None

    # ------------------------------------------------------------------
    # Normal Adaptive Mode
    # ------------------------------------------------------------------

    def compute_timings(self, lane_densities: dict) -> dict:
        """
        Compute green signal time for each lane based on density.

        Args:
            lane_densities: {'lane_1': 20, 'lane_2': 10, ...}

        Returns:
            {'lane_1': 40, 'lane_2': 20, ...}
        """
        self.mode = 'NORMAL'
        self._emergency_lane = None
        total_density = sum(lane_densities.values())

        if total_density == 0:
            n = max(len(lane_densities), 1)
            equal_time = min(self.default_green, self.max_cycle // n)
            self.current_timings = {lane: equal_time for lane in lane_densities}
            return self.current_timings

        timings = {}
        for lane, density in lane_densities.items():
            raw_green = (density / total_density) * self.max_cycle
            green = max(self.min_green, min(self.max_green, raw_green))
            timings[lane] = round(green)

        self.current_timings = timings
        return timings

    # ------------------------------------------------------------------
    # Emergency Override
    # ------------------------------------------------------------------

    def emergency_override(self, emergency_lane: str,
                           all_lanes: list) -> dict:
        """
        Emergency green corridor:
          - Ambulance lane → MAX green
          - All other lanes → 0 (RED)

        Args:
            emergency_lane: Lane containing the ambulance
            all_lanes:      List of all lane names

        Returns:
            Signal timings dict
        """
        self.mode = 'EMERGENCY'
        self._emergency_lane = emergency_lane
        timings = {}
        for lane in all_lanes:
            timings[lane] = self.max_green if lane == emergency_lane else 0
        self.current_timings = timings
        return timings

    # ------------------------------------------------------------------
    # Signal Display State
    # ------------------------------------------------------------------

    def get_signal_display(self) -> dict:
        """
        Get visual signal state for dashboard display.

        Returns dict per lane:
            {'lane_1': {'color': 'GREEN', 'green_time': 60, 'mode': 'EMERGENCY'}, ...}
        """
        display = {}
        if not self.current_timings:
            return display

        if self.mode == 'EMERGENCY':
            for lane, time_val in self.current_timings.items():
                display[lane] = {
                    'color': 'GREEN' if lane == self._emergency_lane else 'RED',
                    'green_time': time_val,
                    'mode': 'EMERGENCY',
                }
            return display

        # NORMAL mode — highest density lane gets green
        max_lane = max(self.current_timings, key=self.current_timings.get)
        for lane, time_val in self.current_timings.items():
            if lane == max_lane:
                color = 'GREEN'
            elif time_val > self.min_green:
                color = 'YELLOW'
            else:
                color = 'RED'
            display[lane] = {
                'color': color,
                'green_time': time_val,
                'mode': 'NORMAL',
            }
        return display

    # ------------------------------------------------------------------
    # Optimization Stats
    # ------------------------------------------------------------------

    def get_optimization_stats(self, lane_densities: dict) -> dict:
        """Calculate optimization metrics vs fixed timing."""
        if not lane_densities:
            return {}

        n = max(len(lane_densities), 1)
        fixed_time = self.max_cycle // n
        adaptive_timings = self.compute_timings(lane_densities)

        total_density = sum(lane_densities.values())
        if total_density == 0:
            return {'improvement': 0}

        fixed_wait = sum(
            density * (self.max_cycle - fixed_time)
            for density in lane_densities.values()
        )
        adaptive_wait = sum(
            density * (self.max_cycle - adaptive_timings[lane])
            for lane, density in lane_densities.items()
        )

        improvement = ((fixed_wait - adaptive_wait) / max(fixed_wait, 1)) * 100

        return {
            'fixed_timing_per_lane': fixed_time,
            'adaptive_timings': adaptive_timings,
            'wait_time_reduction_percent': round(improvement, 1),
            'total_cycle_time': self.max_cycle,
        }
