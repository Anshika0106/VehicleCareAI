"""
Vehicle Telemetry Simulator
Generates realistic vehicle sensor data for testing and prototyping.
"""

import random
import datetime
from typing import Dict, Optional


class VehicleSimulator:
    """Simulates vehicle telemetry data generation."""
    
    def __init__(self, vehicle_id: str = "HERO-MNM-01"):
        """
        Initialize the vehicle simulator.
        
        Args:
            vehicle_id: Unique identifier for the vehicle
        """
        self.vehicle_id = vehicle_id
        self.fault_type: Optional[str] = None
        # State variables for realistic data correlation
        self.last_rpm = 2000.0
        self.last_temp = 90.0
        self.last_vibration = 0.25
        self.last_throttle = 30
        self.last_battery = 14.0
        
    def generate_reading(self) -> Dict:
        """
        Generate a single telemetry reading with realistic sensor values.
        Data is correlated to previous readings for realistic behavior.
        
        Returns:
            Dictionary containing vehicle_id, timestamp, and sensor readings
        """
        # Generate correlated data based on previous state
        # RPM changes gradually (momentum)
        rpm_change = random.uniform(-200, 200)
        base_rpm = max(800, min(3200, self.last_rpm + rpm_change))
        self.last_rpm = base_rpm
        
        # Temperature correlates with RPM (higher RPM = higher temp)
        temp_base = 75 + (base_rpm / 3000) * 30  # 75-105°C range based on RPM
        temp_variation = random.uniform(-3, 3)
        base_temp = max(75, min(110, temp_base + temp_variation))
        self.last_temp = base_temp
        
        # Vibration correlates with RPM and throttle
        vib_base = 0.1 + (base_rpm / 3000) * 0.3  # 0.1-0.4g range
        vib_variation = random.uniform(-0.05, 0.05)
        base_vibration = max(0.05, min(0.45, vib_base + vib_variation))
        self.last_vibration = base_vibration
        
        # Throttle position (more realistic driving pattern)
        throttle_change = random.uniform(-15, 15)
        base_throttle = max(0, min(100, self.last_throttle + throttle_change))
        self.last_throttle = base_throttle
        
        # Battery voltage (relatively stable, slight variation)
        battery_change = random.uniform(-0.1, 0.1)
        base_battery = max(13.0, min(14.8, self.last_battery + battery_change))
        self.last_battery = base_battery
        
        # Apply fault injection if active
        if self.fault_type == "overheat":
            base_temp = random.uniform(120, 140)  # Critical overheating
            self.last_temp = base_temp
        elif self.fault_type == "vibration":
            base_vibration = random.uniform(1.5, 2.5)  # Critical vibration
            self.last_vibration = base_vibration
        
        # Create reading
        reading = {
            "vehicle_id": self.vehicle_id,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sensors": {
                "engine_rpm": round(base_rpm, 2),
                "engine_temp_c": round(base_temp, 2),
                "vibration_level_g": round(base_vibration, 3),
                "throttle_pos_pct": int(base_throttle),
                "battery_voltage_v": round(base_battery, 2)
            }
        }
        
        return reading
    
    def inject_fault(self, fault_type: Optional[str] = None):
        """
        Inject a simulated fault into the sensor readings.
        
        Args:
            fault_type: Type of fault to inject. Options:
                - "overheat": Spikes engine temperature > 120°C
                - "vibration": Spikes vibration > 1.5g
                - None: Clears any active fault (normal operation)
        """
        if fault_type not in [None, "overheat", "vibration"]:
            raise ValueError(f"Unknown fault type: {fault_type}. Use 'overheat', 'vibration', or None")
        
        self.fault_type = fault_type
    
    def clear_fault(self):
        """Clear any active fault and return to normal operation."""
        self.fault_type = None

