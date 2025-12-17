"""
Maintenance Agent
AI agent that interprets anomalies and generates maintenance recommendations.
"""

from typing import Dict, Tuple


def analyze_anomaly(reading: Dict) -> str:
    """
    Analyze an anomalous reading and generate maintenance recommendations.
    
    Args:
        reading: Dictionary containing vehicle_id, timestamp, and sensor readings
        
    Returns:
        Natural language maintenance recommendation
    """
    sensors = reading["sensors"]
    
    # Check for critical vibration
    if sensors["vibration_level_g"] > 1.0:
        return (
            "**CRITICAL: Mechanical looseness detected.**\n\n"
            "**Issue:** Excessive vibration detected ({}g).\n\n"
            "**Recommended Actions:**\n"
            "1. Inspect engine mounts for wear or damage\n"
            "2. Check piston rings and connecting rod bearings\n"
            "3. Examine suspension components\n"
            "4. Verify wheel balance and alignment\n\n"
            "**Quality Alert:** Anomaly pattern sent to manufacturing team for batch analysis."
        ).format(sensors["vibration_level_g"])
    
    # Check for overheating
    if sensors["engine_temp_c"] > 120:
        return (
            "**WARNING: Coolant system failure detected.**\n\n"
            "**Issue:** Engine temperature critically high ({}°C).\n\n"
            "**Recommended Actions:**\n"
            "1. Check radiator fluid levels immediately\n"
            "2. Inspect coolant hoses for leaks or blockages\n"
            "3. Verify thermostat functionality\n"
            "4. Check water pump operation\n"
            "5. Allow engine to cool before inspection\n\n"
            "**Priority:** High - Risk of engine damage if not addressed."
        ).format(sensors["engine_temp_c"])
    
    # Check for battery issues
    if sensors["battery_voltage_v"] < 12.0:
        return (
            "**WARNING: Battery voltage low.**\n\n"
            "**Issue:** Battery voltage below normal range ({}V).\n\n"
            "**Recommended Actions:**\n"
            "1. Test battery charge and health\n"
            "2. Check alternator output\n"
            "3. Inspect battery terminals for corrosion\n"
            "4. Verify electrical system for parasitic drains\n\n"
            "**Priority:** Medium - May cause starting issues."
        ).format(sensors["battery_voltage_v"])
    
    # Check for unusual RPM patterns (throttle malfunction)
    if sensors["engine_rpm"] > 3500 and sensors["throttle_pos_pct"] < 20:
        return (
            "**CRITICAL: Throttle system malfunction detected.**\n\n"
            "**Issue:** High RPM ({}) with low throttle position ({}%). Throttle may be stuck or malfunctioning.\n\n"
            "**Recommended Actions:**\n"
            "1. Check throttle body for sticking or mechanical obstruction\n"
            "2. Inspect idle air control (IAC) valve for proper operation\n"
            "3. Verify throttle position sensor (TPS) calibration\n"
            "4. Check for vacuum leaks in intake system\n"
            "5. Inspect throttle cable for binding or damage\n\n"
            "**Priority:** High - Affects vehicle control and safety."
        ).format(sensors["engine_rpm"], sensors["throttle_pos_pct"])
    
    # Check for engine misfire (low RPM with high vibration)
    if sensors["engine_rpm"] < 1200 and sensors["vibration_level_g"] > 0.6:
        return (
            "**WARNING: Engine misfire detected.**\n\n"
            "**Issue:** Low RPM ({}) with elevated vibration ({}g). Engine may be misfiring.\n\n"
            "**Recommended Actions:**\n"
            "1. Check spark plugs for wear, fouling, or incorrect gap\n"
            "2. Inspect ignition coils and wiring for damage\n"
            "3. Verify fuel injector operation and cleanliness\n"
            "4. Check compression in all cylinders\n"
            "5. Inspect air filter and intake for restrictions\n"
            "6. Verify engine timing and camshaft position\n\n"
            "**Priority:** High - Can cause engine damage and increased emissions."
        ).format(sensors["engine_rpm"], sensors["vibration_level_g"])
    
    # Check for fuel system issues (low RPM despite high throttle)
    if sensors["engine_rpm"] < 1200 and sensors["throttle_pos_pct"] > 40:
        return (
            "**WARNING: Fuel system malfunction detected.**\n\n"
            "**Issue:** Low RPM ({}) despite high throttle position ({}%). Possible fuel delivery problem.\n\n"
            "**Recommended Actions:**\n"
            "1. Check fuel pump pressure and operation\n"
            "2. Inspect fuel filter for clogs or restrictions\n"
            "3. Verify fuel injectors are functioning properly\n"
            "4. Check fuel pressure regulator\n"
            "5. Inspect fuel lines for leaks or blockages\n"
            "6. Verify fuel quality and contamination\n\n"
            "**Priority:** High - Vehicle may stall or fail to start."
        ).format(sensors["engine_rpm"], sensors["throttle_pos_pct"])
    
    # Check for moderate overheating (cooling system issues)
    if 110 < sensors["engine_temp_c"] <= 120:
        return (
            "**WARNING: Cooling system failure detected.**\n\n"
            "**Issue:** Engine temperature elevated ({}°C). Cooling system may be compromised.\n\n"
            "**Recommended Actions:**\n"
            "1. Check coolant level and condition\n"
            "2. Inspect radiator for leaks, clogs, or damage\n"
            "3. Verify radiator fan operation and temperature sensor\n"
            "4. Check water pump for proper circulation\n"
            "5. Inspect thermostat for proper opening/closing\n"
            "6. Check for air bubbles in cooling system\n"
            "7. Verify coolant hoses for leaks or deterioration\n\n"
            "**Priority:** High - May lead to severe engine damage if not addressed."
        ).format(sensors["engine_temp_c"])
    
    # Check for battery voltage issues (more specific)
    if sensors["battery_voltage_v"] < 11.5:
        return (
            "**CRITICAL: Battery failure detected.**\n\n"
            "**Issue:** Battery voltage critically low ({}V). Battery may be failing or charging system malfunctioning.\n\n"
            "**Recommended Actions:**\n"
            "1. Test battery voltage and load capacity immediately\n"
            "2. Check alternator output voltage (should be 13.5-14.5V)\n"
            "3. Inspect battery terminals for corrosion or loose connections\n"
            "4. Verify alternator belt tension and condition\n"
            "5. Check for parasitic electrical drains\n"
            "6. Test battery cells for internal failure\n"
            "7. Consider battery replacement if voltage cannot be restored\n\n"
            "**Priority:** High - Vehicle may not start. Replace battery if necessary."
        ).format(sensors["battery_voltage_v"])
    
    # Generic anomaly (caught by model but no specific pattern)
    return (
        "**ANOMALY DETECTED: Unusual sensor pattern.**\n\n"
        "**Issue:** Multiple sensor readings outside normal parameters.\n\n"
        "**Recommended Actions:**\n"
        "1. Perform comprehensive vehicle inspection\n"
        "2. Review all sensor readings for patterns\n"
        "3. Check for recent maintenance or modifications\n"
        "4. Monitor vehicle behavior over next few trips\n"
        "5. Use diagnostic scanner to check for error codes\n"
        "6. Verify all sensors are calibrated correctly\n\n"
        "**Priority:** Medium - Requires diagnostic investigation."
    )


def get_issue_details(reading: Dict) -> Tuple[str, str, str]:
    """
    Get structured issue details for anomaly notification.
    
    Args:
        reading: Dictionary containing vehicle_id, timestamp, and sensor readings
        
    Returns:
        Tuple of (issue_title, issue_description, recommended_action)
    """
    sensors = reading["sensors"]
    
    # Check for critical vibration
    if sensors["vibration_level_g"] > 1.0:
        return (
            "Mechanical Looseness Detected",
            f"The predictive model analysis indicates excessive vibration levels ({sensors['vibration_level_g']:.2f}g) in the vehicle, suggesting potential mechanical looseness in engine mounts, suspension components, or wheel assemblies.",
            "Schedule a service appointment immediately to inspect engine mounts, piston rings, connecting rod bearings, and suspension components."
        )
    
    # Check for overheating
    if sensors["engine_temp_c"] > 120:
        return (
            "Coolant System Failure",
            f"The predictive model analysis indicates critically high engine temperature ({sensors['engine_temp_c']:.1f}°C), suggesting coolant system failure or thermostat malfunction.",
            "Schedule a service appointment immediately to inspect radiator fluid levels, coolant hoses, thermostat, and water pump operation."
        )
    
    # Check for battery issues
    if sensors["battery_voltage_v"] < 12.0:
        return (
            "Battery Health Deterioration",
            f"The predictive model analysis indicates an increasingly low battery voltage ({sensors['battery_voltage_v']:.2f}V) in the primary vehicle battery, suggesting failure potential within the next 4-6 weeks.",
            "Schedule a service appointment immediately to test battery capacity and replacement if necessary."
        )
    
    # Check for unusual RPM patterns (throttle malfunction)
    if sensors["engine_rpm"] > 3500 and sensors["throttle_pos_pct"] < 20:
        return (
            "Throttle System Malfunction",
            f"The predictive model analysis indicates high RPM ({sensors['engine_rpm']:.0f}) with low throttle position ({sensors['throttle_pos_pct']}%), suggesting throttle body sticking or malfunction.",
            "Schedule a service appointment immediately to inspect throttle body, idle air control valve, and throttle position sensor calibration."
        )
    
    # Check for engine misfire
    if sensors["engine_rpm"] < 1200 and sensors["vibration_level_g"] > 0.6:
        return (
            "Engine Misfire Detected",
            f"The predictive model analysis indicates low RPM ({sensors['engine_rpm']:.0f}) with elevated vibration ({sensors['vibration_level_g']:.2f}g), suggesting engine cylinder misfiring.",
            "Schedule a service appointment immediately to check spark plugs, ignition coils, fuel injectors, and engine compression."
        )
    
    # Check for fuel system issues
    if sensors["engine_rpm"] < 1200 and sensors["throttle_pos_pct"] > 40:
        return (
            "Fuel System Malfunction",
            f"The predictive model analysis indicates low RPM ({sensors['engine_rpm']:.0f}) despite high throttle position ({sensors['throttle_pos_pct']}%), suggesting fuel delivery problems.",
            "Schedule a service appointment immediately to check fuel pump pressure, fuel filter, and fuel injector operation."
        )
    
    # Check for moderate overheating (cooling system issues)
    if 110 < sensors["engine_temp_c"] <= 120:
        return (
            "Cooling System Failure",
            f"The predictive model analysis indicates elevated engine temperature ({sensors['engine_temp_c']:.1f}°C), suggesting cooling system compromise or radiator blockage.",
            "Schedule a service appointment immediately to check coolant levels, radiator condition, fan operation, and water pump functionality."
        )
    
    # Check for critical battery voltage
    if sensors["battery_voltage_v"] < 11.5:
        return (
            "Battery Failure Critical",
            f"The predictive model analysis indicates critically low battery voltage ({sensors['battery_voltage_v']:.2f}V), suggesting immediate battery or charging system failure.",
            "Schedule a service appointment immediately to test battery and alternator, and replace battery if necessary."
        )
    
    # Generic anomaly
    return (
        "Unusual Sensor Pattern Detected",
        "The predictive model analysis indicates multiple sensor readings outside normal parameters, suggesting potential component degradation or sensor calibration issues.",
        "Schedule a service appointment to perform comprehensive vehicle inspection and diagnostic scan."
    )


def get_severity_level(reading: Dict) -> str:
    """
    Determine the severity level of an anomaly.
    
    Args:
        reading: Dictionary containing vehicle_id, timestamp, and sensor readings
        
    Returns:
        Severity level: "Critical", "Major", or "Minor"
    """
    sensors = reading["sensors"]
    
    # Critical conditions
    if (sensors["vibration_level_g"] > 1.0 or 
        sensors["engine_temp_c"] > 120 or 
        sensors["battery_voltage_v"] < 11.5 or
        (sensors["engine_rpm"] > 3500 and sensors["throttle_pos_pct"] < 20)):
        return "Critical"
    
    # Major conditions
    if (sensors["battery_voltage_v"] < 12.0 or
        sensors["engine_temp_c"] > 110 or
        sensors["vibration_level_g"] > 0.6 or
        (sensors["engine_rpm"] < 1200 and sensors["throttle_pos_pct"] > 40)):
        return "Major"
    
    # Minor/default
    return "Minor"


def calculate_health_score(reading: Dict) -> int:
    """
    Calculate overall vehicle health score (0-100).
    
    Args:
        reading: Dictionary containing vehicle_id, timestamp, and sensor readings
        
    Returns:
        Health score (0-100, where 100 is perfect health)
    """
    sensors = reading["sensors"]
    score = 100
    
    # Deduct points for temperature issues
    if sensors["engine_temp_c"] > 105:
        score -= min(30, (sensors["engine_temp_c"] - 105) * 2)
    
    # Deduct points for vibration issues
    if sensors["vibration_level_g"] > 0.4:
        score -= min(25, (sensors["vibration_level_g"] - 0.4) * 40)
    
    # Deduct points for battery issues
    if sensors["battery_voltage_v"] < 13.5:
        score -= min(20, (13.5 - sensors["battery_voltage_v"]) * 10)
    elif sensors["battery_voltage_v"] > 14.5:
        score -= min(15, (sensors["battery_voltage_v"] - 14.5) * 10)
    
    # Deduct points for RPM issues
    if sensors["engine_rpm"] > 3000:
        score -= min(15, (sensors["engine_rpm"] - 3000) * 0.01)
    elif sensors["engine_rpm"] < 800:
        score -= min(15, (800 - sensors["engine_rpm"]) * 0.02)
    
    return max(0, int(score))


def get_predicted_issue(reading: Dict) -> str:
    """
    Get a short description of the predicted issue.
    
    Args:
        reading: Dictionary containing vehicle_id, timestamp, and sensor readings
        
    Returns:
        Short issue description
    """
    sensors = reading["sensors"]
    
    if sensors["battery_voltage_v"] < 12.5:
        return "Low Battery Voltage detected in near future"
    elif sensors["engine_temp_c"] > 105:
        return "Engine Overheating risk detected"
    elif sensors["vibration_level_g"] > 0.4:
        return "Mechanical vibration issue detected"
    elif sensors["engine_rpm"] > 3000:
        return "Engine stress detected"
    else:
        return "Minor sensor anomalies detected"


def get_risk_level(reading: Dict) -> str:
    """
    Get the risk level for the predicted issue.
    
    Args:
        reading: Dictionary containing vehicle_id, timestamp, and sensor readings
        
    Returns:
        Risk level: "High", "Medium", or "Low"
    """
    severity = get_severity_level(reading)
    
    if severity == "Critical":
        return "High"
    elif severity == "Major":
        return "High"
    else:
        return "Medium"

