"""
============================================
Density Score Calculator
============================================
Calculates traffic density scores based on
vehicle counts and type-based weights.

Density = sum(vehicle_count × vehicle_weight)

Example:
  10 cars (×2) + 2 buses (×5) + 1 truck (×6) = 36
"""


class DensityCalculator:
    """
    Calculates weighted traffic density scores.
    
    Vehicle Type Weights:
        Motorcycle = 1.0  (small footprint)
        Vehicle    = 2.0  (standard cars)
        Bus        = 5.0  (large, blocks lane)
        Truck      = 6.0  (largest)
        Ambulance  = 0.0  (excluded from density)
    """
    
    def __init__(self, weights=None):
        self.weights = weights or {
            'Vehicle': 2.0,
            'Bus': 5.0,
            'Truck': 6.0,
            'Motorcycle': 1.0,
            'Ambulance': 0.0,
        }
    
    def calculate(self, vehicle_counts: dict) -> float:
        """
        Calculate total density score from vehicle counts.
        
        Args:
            vehicle_counts: {'car': 10, 'bus': 2, 'truck': 1, ...}
        
        Returns:
            Weighted density score (float)
        """
        density = 0.0
        for vehicle_type, count in vehicle_counts.items():
            weight = self.weights.get(vehicle_type, 1.0)
            density += count * weight
        return density
    
    def calculate_per_lane(self, lane_counts: dict) -> dict:
        """
        Calculate density for multiple lanes.
        
        Args:
            lane_counts: {
                'lane_1': {'car': 5, 'bus': 1},
                'lane_2': {'car': 3, 'truck': 2},
            }
        
        Returns:
            {'lane_1': 15.0, 'lane_2': 18.0}
        """
        return {lane: self.calculate(counts) 
                for lane, counts in lane_counts.items()}
    
    def congestion_level(self, density: float) -> str:
        """
        Classify congestion level from density score.
        
        Returns: 'LOW', 'MODERATE', 'HIGH', or 'CRITICAL'
        """
        if density < 15:
            return 'LOW'
        elif density < 30:
            return 'MODERATE'
        elif density < 50:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    def get_summary(self, vehicle_counts: dict) -> dict:
        """Get full density analysis summary."""
        density = self.calculate(vehicle_counts)
        total_vehicles = sum(vehicle_counts.values())
        return {
            'total_vehicles': total_vehicles,
            'density_score': round(density, 1),
            'congestion_level': self.congestion_level(density),
            'breakdown': {k: v for k, v in vehicle_counts.items() if v > 0},
        }
