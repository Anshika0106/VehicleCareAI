"""
Vehicle Telemetry Simulator
Generates realistic vehicle sensor data for testing and prototyping.
All sensors are interconnected to simulate realistic physical relationships.
"""

import random
import datetime
from typing import Dict, Optional


class VehicleSimulator:
    """
    Simulates vehicle telemetry data generation with physically realistic
    interconnected sensor values.
    
    Physical Relationships:
    - Throttle Position → drives RPM (more throttle = higher RPM)
    - RPM over time → affects Temperature (high RPM heats up, idle cools down)
    - RPM → affects Vibration (higher RPM = more vibration)
    - Electrical Load → affects Battery Voltage (slight drain under load)
    """
    
    def __init__(self, vehicle_id: str = "HERO-MNM-01"):
        """
        Initialize the vehicle simulator.
        
        Args:
            vehicle_id: Unique identifier for the vehicle
        """
        self.vehicle_id = vehicle_id
        self.fault_type: Optional[str] = None
        
        # State variables for realistic data correlation
        self.throttle = 25.0  # Driver input (0-100%)
        self.rpm = 1000.0  # Engine RPM (follows throttle)
        self.temperature = 85.0  # Engine temp (follows RPM over time)
        self.vibration = 0.15  # Vibration (follows RPM)
        self.battery = 14.0  # Battery voltage
        
        # Target temperature for cooling (ambient/idle temp)
        self.idle_temp = 82.0
        
    def generate_reading(self) -> Dict:
        """
        Generate a single telemetry reading with realistic interconnected sensor values.
        
        The chain of causation:
        1. Throttle changes (simulates driver input)
        2. RPM responds to throttle (with some lag/smoothing)
        3. Temperature responds to RPM (heat buildup/cooling)
        4. Vibration responds to RPM
        5. Battery responds to electrical load
        
        Returns:
            Dictionary containing vehicle_id, timestamp, and sensor readings
        """
        # ============================================
        # STEP 1: Throttle Position (Driver Input)
        # ============================================
        # Simulate realistic driving: gradual changes with occasional quick moves
        if random.random() < 0.1:  # 10% chance of bigger throttle change
            throttle_change = random.uniform(-20, 20)
        else:
            throttle_change = random.uniform(-8, 8)
        
        self.throttle = max(0, min(100, self.throttle + throttle_change))
        
        # ============================================
        # STEP 2: RPM responds to Throttle
        # ============================================
        # Target RPM based on throttle position
        # Idle (0% throttle) = ~800 RPM, Full throttle (100%) = ~3500 RPM
        target_rpm = 800 + (self.throttle / 100) * 2700
        
        # RPM moves toward target with some lag (engine response time)
        # Add small random variation for realism
        rpm_response_rate = 0.3  # How quickly RPM responds (0-1)
        rpm_noise = random.uniform(-50, 50)
        self.rpm = self.rpm + (target_rpm - self.rpm) * rpm_response_rate + rpm_noise
        self.rpm = max(750, min(3500, self.rpm))
        
        # ============================================
        # STEP 3: Temperature responds to RPM (thermal dynamics)
        # ============================================
        # Higher RPM generates more heat, idle/low RPM allows cooling
        # Heat generation is proportional to RPM^2 (power output)
        heat_generation = ((self.rpm - 800) / 2700) ** 1.5 * 0.8  # 0 to 0.8 scale
        
        # Cooling is proportional to how much above ambient we are
        cooling_rate = (self.temperature - self.idle_temp) * 0.05
        
        # Net temperature change
        temp_change = heat_generation - cooling_rate + random.uniform(-0.5, 0.5)
        self.temperature = self.temperature + temp_change
        
        # Clamp to realistic range (ambient to slightly warm)
        self.temperature = max(75, min(105, self.temperature))
        
        # ============================================
        # STEP 4: Vibration responds to RPM
        # ============================================
        # Base vibration increases with RPM
        # Also slight increase with throttle (engine load)
        base_vib = 0.08 + (self.rpm / 3500) * 0.25  # 0.08g at idle, up to 0.33g at high RPM
        load_vib = (self.throttle / 100) * 0.05  # Additional vibration from load
        vib_noise = random.uniform(-0.02, 0.02)
        
        self.vibration = base_vib + load_vib + vib_noise
        self.vibration = max(0.05, min(0.40, self.vibration))
        
        # ============================================
        # STEP 5: Battery responds to electrical load
        # ============================================
        # Higher RPM = alternator charging better
        # Higher electrical load (throttle as proxy) = slight drain
        alternator_output = 13.5 + (self.rpm / 3500) * 1.3  # 13.5V at idle, 14.8V at high RPM
        electrical_load = (self.throttle / 100) * 0.3  # Load from accessories
        battery_noise = random.uniform(-0.05, 0.05)
        
        target_battery = alternator_output - electrical_load
        self.battery = self.battery + (target_battery - self.battery) * 0.2 + battery_noise
        self.battery = max(13.2, min(14.8, self.battery))
        
        # Start with the interconnected normal values
        final_rpm = self.rpm
        final_temp = self.temperature
        final_vibration = self.vibration
        final_throttle = self.throttle
        final_battery = self.battery
        
        # Apply fault injection if active (override specific values)
        if self.fault_type == "overheat":
            final_temp = random.uniform(120, 140)  # Critical overheating
        elif self.fault_type == "vibration":
            final_vibration = random.uniform(1.5, 2.5)  # Critical vibration
        elif self.fault_type == "battery_failure":
            final_battery = random.uniform(11.0, 11.8)  # Low battery voltage
        elif self.fault_type == "throttle_malfunction":
            # High RPM with low throttle (throttle stuck or malfunctioning)
            final_rpm = random.uniform(3500, 4000)
            final_throttle = random.uniform(5, 15)  # Low throttle despite high RPM
        elif self.fault_type == "engine_misfire":
            # Irregular RPM patterns (engine misfiring)
            final_rpm = random.uniform(800, 1200)  # Low, unstable RPM
            final_vibration = random.uniform(0.6, 0.9)  # Increased vibration
            final_temp = random.uniform(70, 85)  # Lower temp due to misfire
        elif self.fault_type == "fuel_system":
            # Fuel system issues - affects RPM and throttle response
            final_rpm = random.uniform(600, 1000)  # Low RPM, struggling
            final_throttle = random.uniform(40, 60)  # High throttle but low RPM
            final_temp = random.uniform(65, 80)  # Lower temp
        elif self.fault_type == "cooling_system":
            # Cooling system failure - gradual overheating
            final_temp = random.uniform(115, 125)  # Moderate overheating
            # Keep RPM from interconnected system for realism
        
        # Create reading with interconnected values
        reading = {
            "vehicle_id": self.vehicle_id,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sensors": {
                "engine_rpm": round(final_rpm, 2),
                "engine_temp_c": round(final_temp, 2),
                "vibration_level_g": round(final_vibration, 3),
                "throttle_pos_pct": int(final_throttle),
                "battery_voltage_v": round(final_battery, 2)
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
                - "battery_failure": Low battery voltage < 12V
                - "throttle_malfunction": High RPM with low throttle position
                - "engine_misfire": Irregular, low RPM with increased vibration
                - "fuel_system": Low RPM despite high throttle (fuel delivery issues)
                - "cooling_system": Moderate overheating (cooling system failure)
                - None: Clears any active fault (normal operation)
        """
        valid_faults = [None, "overheat", "vibration", "battery_failure", 
                       "throttle_malfunction", "engine_misfire", 
                       "fuel_system", "cooling_system"]
        if fault_type not in valid_faults:
            raise ValueError(f"Unknown fault type: {fault_type}. Valid options: {valid_faults}")
        
        self.fault_type = fault_type
    
    def clear_fault(self):
        """Clear any active fault and return to normal operation."""
        self.fault_type = None

