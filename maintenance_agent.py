"""
Maintenance Agent
AI agent that interprets anomalies and generates maintenance recommendations.
"""

from typing import Dict


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
    
    # Check for unusual RPM patterns
    if sensors["engine_rpm"] > 3500 and sensors["throttle_pos_pct"] < 20:
        return (
            "**WARNING: Unusual engine behavior detected.**\n\n"
            "**Issue:** High RPM ({}) with low throttle position ({}%).\n\n"
            "**Recommended Actions:**\n"
            "1. Check throttle body for sticking or malfunction\n"
            "2. Inspect idle air control valve\n"
            "3. Verify throttle position sensor calibration\n"
            "4. Check for vacuum leaks\n\n"
            "**Priority:** Medium - May indicate throttle system issues."
        ).format(sensors["engine_rpm"], sensors["throttle_pos_pct"])
    
    # Generic anomaly (caught by model but no specific pattern)
    return (
        "**ANOMALY DETECTED: Unusual sensor pattern.**\n\n"
        "**Issue:** Multiple sensor readings outside normal parameters.\n\n"
        "**Recommended Actions:**\n"
        "1. Perform comprehensive vehicle inspection\n"
        "2. Review all sensor readings for patterns\n"
        "3. Check for recent maintenance or modifications\n"
        "4. Monitor vehicle behavior over next few trips\n\n"
        "**Priority:** Medium - Requires diagnostic investigation."
    )

