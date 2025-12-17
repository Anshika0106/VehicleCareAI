"""
VehicleCare AI - Predictive Maintenance Dashboard
Streamlit application for real-time vehicle monitoring and anomaly detection.
"""

import streamlit as st
import time
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vehicle_sim import VehicleSimulator
from anomaly_model import AnomalyDetector
from maintenance_agent import (
    analyze_anomaly, 
    get_issue_details, 
    get_severity_level,
    calculate_health_score,
    get_predicted_issue,
    get_risk_level
)

# Page configuration
st.set_page_config(
    page_title="VehicleCare AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "simulator" not in st.session_state:
    st.session_state.simulator = VehicleSimulator()
if "detector" not in st.session_state:
    st.session_state.detector = AnomalyDetector()
elif not hasattr(st.session_state.detector, 'sync_history'):
    # Recreate detector if it's an old version without sync_history method
    # Preserve training state if model was already trained
    was_trained = st.session_state.detector.is_trained
    st.session_state.detector = AnomalyDetector()
    if was_trained:
        # Retrain the model
        st.session_state.model_trained = False
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "readings_history" not in st.session_state:
    st.session_state.readings_history = []
if "anomalies_detected" not in st.session_state:
    st.session_state.anomalies_detected = []
if "auto_update" not in st.session_state:
    st.session_state.auto_update = True  # Start with auto-update enabled
if "update_interval" not in st.session_state:
    st.session_state.update_interval = 5  # Default to 5 seconds
if "last_update_time" not in st.session_state:
    st.session_state.last_update_time = time.time()
if "current_page" not in st.session_state:
    st.session_state.current_page = "dashboard"  # dashboard, issue_details, schedule_service, confirmation
if "current_issue" not in st.session_state:
    st.session_state.current_issue = None
if "appointments" not in st.session_state:
    st.session_state.appointments = []
if "latest_appointment" not in st.session_state:
    st.session_state.latest_appointment = None
if "show_notification" not in st.session_state:
    st.session_state.show_notification = False

# Check if model is already trained (loaded from disk or in session)
if not st.session_state.model_trained:
    # Check if detector already loaded models from disk
    if st.session_state.detector.is_trained:
        st.session_state.model_trained = True
        st.success("✓ Loaded pre-trained model from disk.")
    else:
        # Only train if no saved model exists
        with st.spinner("Training anomaly detection model on normal vehicle data... (This only happens once)"):
            st.session_state.detector.train_initial_model(n_samples=1000)
            st.session_state.model_trained = True
            st.success("Model trained and saved successfully!")


# Helper functions for page rendering
def render_issue_details_page():
    """Render the Issue Detected page based on wireframe."""
    st.markdown(
        """
        <style>
        .issue-header {
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 30px;
        }
        .issue-box {
            border: 2px solid #333;
            padding: 20px;
            margin: 20px 0;
            background-color: #f9f9f9;
        }
        .issue-title {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .issue-description {
            color: #666;
            line-height: 1.6;
        }
        .severity-label {
            text-align: right;
            font-weight: bold;
            font-size: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Back button
    if st.button("← Back"):
        st.session_state.current_page = "dashboard"
        st.session_state.show_notification = False
        st.rerun()
    
    st.markdown('<div class="issue-header">VehicleCare AI</div>', unsafe_allow_html=True)
    
    if st.session_state.current_issue:
        issue = st.session_state.current_issue
        reading = issue["reading"]
        
        # Get structured issue details
        issue_title, issue_description, recommended_action = get_issue_details(reading)
        severity = get_severity_level(reading)
        
        # Issue Detected section
        st.markdown('<div class="issue-box">', unsafe_allow_html=True)
        st.markdown('<div class="issue-title">Issue Detected</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="issue-title" style="margin-top: 15px;">{issue_title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="issue-description">{issue_description}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Issue Details section
        st.markdown('<div class="issue-box">', unsafe_allow_html=True)
        st.markdown('<div class="issue-title">Issue Details</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Severity Level:**")
        with col2:
            st.markdown(f'<div class="severity-label">{severity}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommended Action section
        st.markdown('<div class="issue-box">', unsafe_allow_html=True)
        st.markdown('<div class="issue-title">Recommended Action</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="issue-description">{recommended_action}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Schedule Service button
        if st.button("Schedule Service", type="primary", use_container_width=True):
            st.session_state.current_page = "schedule_service"
            st.rerun()


def render_schedule_service_page():
    """Render the Schedule Service page."""
    st.markdown('<div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 30px;">VehicleCare AI</div>', unsafe_allow_html=True)
    
    # Back button
    if st.button("← Back"):
        st.session_state.current_page = "issue_details"
        st.rerun()
    
    st.markdown("### Schedule Service Appointment")
    
    if st.session_state.current_issue:
        issue = st.session_state.current_issue
        reading = issue["reading"]
        issue_title, _, _ = get_issue_details(reading)
        
        # Service center selection
        service_centers = [
            "VehicleCare Certified Center - Downtown",
            "VehicleCare Certified Center - Uptown",
            "VehicleCare Certified Center - Westside",
            "VehicleCare Certified Center - Eastside"
        ]
        
        selected_center = st.selectbox("Select Service Center", service_centers)
        
        # Service type (auto-fill based on issue)
        service_types = {
            "Battery Health Deterioration": "Battery Diagnosis & Replacement",
            "Battery Failure Critical": "Battery Diagnosis & Replacement",
            "Coolant System Failure": "Cooling System Inspection & Repair",
            "Cooling System Failure": "Cooling System Inspection & Repair",
            "Mechanical Looseness Detected": "Vibration Diagnosis & Repair",
            "Engine Misfire Detected": "Engine Inspection & Repair",
            "Throttle System Malfunction": "Throttle System Repair",
            "Fuel System Malfunction": "Fuel System Inspection & Repair"
        }
        
        # Create options list with unique services
        service_options = sorted(list(set(service_types.values()))) + ["General Inspection & Diagnosis"]
        
        # Get default service based on issue
        default_service = service_types.get(issue_title, "General Inspection & Diagnosis")
        
        # Find index of default service in options
        try:
            default_index = service_options.index(default_service)
        except ValueError:
            default_index = len(service_options) - 1  # Default to "General Inspection & Diagnosis"
        
        service_type = st.selectbox("Service Type", service_options, index=default_index)
        
        # Date selection
        min_date = datetime.now().date() + timedelta(days=1)
        max_date = datetime.now().date() + timedelta(days=30)
        selected_date = st.date_input("Select Date", min_value=min_date, max_value=max_date, value=min_date)
        
        # Time selection
        time_slots = [
            "08:00 AM", "09:00 AM", "10:00 AM", "11:00 AM",
            "12:00 PM", "01:00 PM", "02:00 PM", "03:00 PM",
            "04:00 PM", "05:00 PM"
        ]
        selected_time = st.selectbox("Select Time", time_slots)
        
        # Customer information
        st.markdown("### Customer Information")
        customer_name = st.text_input("Name", value="John Doe")
        customer_phone = st.text_input("Phone", value="+1 (555) 123-4567")
        customer_email = st.text_input("Email", value="john.doe@example.com")
        
        # Confirm booking button
        if st.button("Confirm Booking", type="primary", use_container_width=True):
            # Create appointment
            appointment = {
                "service_center": selected_center,
                "service_type": service_type,
                "date": selected_date,
                "time": selected_time,
                "customer_name": customer_name,
                "customer_phone": customer_phone,
                "customer_email": customer_email,
                "issue": issue_title,
                "status": "Confirmed",
                "created_at": datetime.now()
            }
            
            st.session_state.appointments.append(appointment)
            st.session_state.latest_appointment = appointment
            st.session_state.current_page = "confirmation"
            st.rerun()


def render_confirmation_page():
    """Render the Appointment Confirmation page based on wireframe."""
    st.markdown(
        """
        <style>
        .confirmation-header {
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 30px;
        }
        .status-box {
            border: 2px solid #333;
            padding: 20px;
            margin: 20px 0;
            background-color: #e8e8e8;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
        }
        .details-box {
            border: 2px solid #333;
            padding: 20px;
            margin: 20px 0;
        }
        .detail-row {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
        }
        .detail-label {
            color: #666;
        }
        .detail-value {
            font-weight: bold;
            text-align: right;
        }
        .info-box {
            background-color: #f0f0f0;
            padding: 15px;
            margin: 20px 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="confirmation-header">Appointment Confirmation</div>', unsafe_allow_html=True)
    
    if st.session_state.latest_appointment:
        appointment = st.session_state.latest_appointment
        
        # Status box
        st.markdown(f'<div class="status-box">Status: {appointment["status"]}</div>', unsafe_allow_html=True)
        
        # Appointment Details
        st.markdown('<div class="details-box">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 20px; font-weight: bold; margin-bottom: 15px;">Appointment Details</div>', unsafe_allow_html=True)
        
        # Service Center
        st.markdown(
            f'<div class="detail-row">'
            f'<span class="detail-label">Service Center:</span>'
            f'<span class="detail-value">{appointment["service_center"]}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        # Service Type
        st.markdown(
            f'<div class="detail-row">'
            f'<span class="detail-label">Service Type:</span>'
            f'<span class="detail-value">{appointment["service_type"]}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        # Date
        date_str = appointment["date"].strftime("%A, %B %d, %Y")
        st.markdown(
            f'<div class="detail-row">'
            f'<span class="detail-label">Date:</span>'
            f'<span class="detail-value">{date_str}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        # Time
        st.markdown(
            f'<div class="detail-row">'
            f'<span class="detail-label">Time:</span>'
            f'<span class="detail-value">{appointment["time"]}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Information box
        st.markdown(
            '<div class="info-box">Your appointment has been confirmed. Please arrive 10 minutes early.</div>',
            unsafe_allow_html=True
        )
        
        # Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("View Appointment", use_container_width=True):
                st.session_state.current_page = "appointments"
                st.rerun()
        
        with col2:
            if st.button("Back to Dashboard", use_container_width=True):
                st.session_state.current_page = "dashboard"
                st.session_state.show_notification = False
                st.rerun()


def render_vehicle_health_dashboard():
    """Render the Vehicle Health Dashboard view based on wireframe."""
    st.markdown(
        """
        <style>
        .health-header {
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 30px;
        }
        .health-box {
            border: 2px solid #333;
            padding: 20px;
            margin: 20px 0;
        }
        .health-title {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 15px;
        }
        .health-detail {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
        }
        .health-label {
            color: #666;
        }
        .health-value {
            font-weight: bold;
            text-align: right;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="health-header">VehicleCare AI</div>', unsafe_allow_html=True)
    
    # Get latest reading
    if "latest_reading" in st.session_state:
        latest = st.session_state.latest_reading
    elif st.session_state.readings_history:
        latest = st.session_state.readings_history[-1]
    else:
        latest = None
    
    if latest:
        # Calculate health metrics
        health_score = calculate_health_score(latest)
        predicted_issue = get_predicted_issue(latest)
        risk_level = get_risk_level(latest)
        
        # Vehicle Health Dashboard box
        st.markdown('<div class="health-box">', unsafe_allow_html=True)
        st.markdown('<div class="health-title">Vehicle Health Dashboard</div>', unsafe_allow_html=True)
        
        # Vehicle ID
        vehicle_id = latest.get("vehicle_id", "VIN: 1FA6P00000005721")
        st.markdown(
            f'<div class="health-detail">'
            f'<span class="health-label">Vehicle ID:</span>'
            f'<span class="health-value">{vehicle_id}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        # Health Score
        st.markdown(
            f'<div class="health-detail">'
            f'<span class="health-label">Health Score:</span>'
            f'<span class="health-value">{health_score}%</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        # Predicted Issue
        st.markdown(
            f'<div class="health-detail">'
            f'<span class="health-label">Predicted Issue:</span>'
            f'<span class="health-value">{predicted_issue}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        # Risk Level
        st.markdown(
            f'<div class="health-detail">'
            f'<span class="health-label">Risk Level:</span>'
            f'<span class="health-value">{risk_level}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # View Details button
        if st.button("View Details", type="primary", use_container_width=True):
            # Check if there's a current issue to view
            if st.session_state.anomalies_detected:
                st.session_state.current_issue = st.session_state.anomalies_detected[-1]
                st.session_state.current_page = "issue_details"
            else:
                # Switch to full dashboard view
                st.session_state.current_page = "dashboard"
            st.rerun()
    else:
        st.info("No vehicle data available. Enable Auto Update to start monitoring.")


def render_appointments_page():
    """Render the appointments list page."""
    st.markdown('<div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 30px;">VehicleCare AI</div>', unsafe_allow_html=True)
    
    # Back button
    if st.button("← Back to Dashboard"):
        st.session_state.current_page = "dashboard"
        st.rerun()
    
    st.markdown("### My Appointments")
    
    if st.session_state.appointments:
        for idx, appointment in enumerate(reversed(st.session_state.appointments)):
            with st.expander(f"Appointment {len(st.session_state.appointments) - idx} - {appointment['date'].strftime('%B %d, %Y')}"):
                st.markdown(f"**Status:** {appointment['status']}")
                st.markdown(f"**Service Center:** {appointment['service_center']}")
                st.markdown(f"**Service Type:** {appointment['service_type']}")
                st.markdown(f"**Date:** {appointment['date'].strftime('%A, %B %d, %Y')}")
                st.markdown(f"**Time:** {appointment['time']}")
                st.markdown(f"**Issue:** {appointment['issue']}")
                st.markdown(f"**Customer:** {appointment['customer_name']}")
                st.markdown(f"**Phone:** {appointment['customer_phone']}")
                st.markdown(f"**Email:** {appointment['customer_email']}")
    else:
        st.info("No appointments scheduled yet.")

# Sidebar
with st.sidebar:
    st.title("VehicleCare AI")
    st.markdown("---")
    
    st.subheader("Vehicle Control")
    vehicle_id = st.text_input("Vehicle ID", value="HERO-MNM-01")
    st.session_state.simulator.vehicle_id = vehicle_id
    
    st.markdown("---")
    
    st.subheader("Fault Simulation")
    fault_type = st.selectbox(
        "Simulate Component Failure",
        ["None", "Overheat", "Vibration", "Battery Failure", 
         "Throttle Malfunction", "Engine Misfire", 
         "Fuel System Issue", "Cooling System Failure"],
        index=0
    )
    
    if fault_type == "Overheat":
        st.session_state.simulator.inject_fault("overheat")
        st.warning("⚠️ Overheating fault active")
    elif fault_type == "Vibration":
        st.session_state.simulator.inject_fault("vibration")
        st.warning("⚠️ Vibration fault active")
    elif fault_type == "Battery Failure":
        st.session_state.simulator.inject_fault("battery_failure")
        st.warning("⚠️ Battery failure active")
    elif fault_type == "Throttle Malfunction":
        st.session_state.simulator.inject_fault("throttle_malfunction")
        st.warning("⚠️ Throttle malfunction active")
    elif fault_type == "Engine Misfire":
        st.session_state.simulator.inject_fault("engine_misfire")
        st.warning("⚠️ Engine misfire active")
    elif fault_type == "Fuel System Issue":
        st.session_state.simulator.inject_fault("fuel_system")
        st.warning("⚠️ Fuel system issue active")
    elif fault_type == "Cooling System Failure":
        st.session_state.simulator.inject_fault("cooling_system")
        st.warning("⚠️ Cooling system failure active")
    else:
        st.session_state.simulator.inject_fault(None)
        st.success("✓ Normal operation")
    
    st.markdown("---")
    
    st.subheader("Dashboard Controls")
    auto_update = st.checkbox("Auto Update", value=st.session_state.auto_update)
    st.session_state.auto_update = auto_update
    
    # Update interval dropdown
    interval_options = {
        "5 seconds": 5,
        "10 seconds": 10,
        "30 seconds": 30,
        "1 minute": 60,
        "2 minutes": 120,
        "5 minutes": 300,
        "10 minutes": 600
    }
    
    # Find the index of current interval
    interval_list = list(interval_options.items())
    current_index = 0
    for idx, (label, seconds) in enumerate(interval_list):
        if seconds == st.session_state.update_interval:
            current_index = idx
            break
    
    selected_interval = st.selectbox(
        "Update Interval",
        options=list(interval_options.keys()),
        index=current_index
    )
    st.session_state.update_interval = interval_options[selected_interval]
    
    if st.button("Generate New Reading"):
        # Sync detector history before detection (if method exists)
        if hasattr(st.session_state.detector, 'sync_history'):
            st.session_state.detector.sync_history(st.session_state.readings_history)
        
        reading = st.session_state.simulator.generate_reading()
        anomaly = st.session_state.detector.detect_anomaly(reading)
        score = st.session_state.detector.get_anomaly_score(reading)
        
        reading["anomaly"] = anomaly
        reading["anomaly_score"] = score
        st.session_state.readings_history.append(reading)
        
        if anomaly == -1:
            recommendation = analyze_anomaly(reading)
            anomaly_data = {
                "timestamp": reading["timestamp"],
                "reading": reading,
                "recommendation": recommendation
            }
            st.session_state.anomalies_detected.append(anomaly_data)
            
            # Set current issue and show notification
            st.session_state.current_issue = anomaly_data
            st.session_state.show_notification = True
    
    if st.button("Clear History"):
        st.session_state.readings_history = []
        st.session_state.anomalies_detected = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("**Status:**")
    st.info(f"Readings: {len(st.session_state.readings_history)}")
    st.info(f"Anomalies: {len(st.session_state.anomalies_detected)}")
    
    st.markdown("---")
    st.markdown("**Quick Access:**")
    if st.button("Vehicle Health Dashboard", use_container_width=True):
        st.session_state.current_page = "health_dashboard"
        st.rerun()
    
    if st.session_state.appointments:
        if st.button(f"View Appointments ({len(st.session_state.appointments)})", use_container_width=True):
            st.session_state.current_page = "appointments"
            st.rerun()

# Main dashboard - Page routing
# Check if there's a notification to show
if st.session_state.show_notification and st.session_state.current_page == "dashboard":
    # Auto-navigate to issue details page when anomaly detected
    st.session_state.current_page = "issue_details"
    st.rerun()

# Route to appropriate page
if st.session_state.current_page == "issue_details":
    render_issue_details_page()
    st.stop()
elif st.session_state.current_page == "schedule_service":
    render_schedule_service_page()
    st.stop()
elif st.session_state.current_page == "confirmation":
    render_confirmation_page()
    st.stop()
elif st.session_state.current_page == "health_dashboard":
    render_vehicle_health_dashboard()
    st.stop()
elif st.session_state.current_page == "appointments":
    render_appointments_page()
    st.stop()

# Default: Full dashboard
st.title("VehicleCare AI - Predictive Maintenance Dashboard")
st.markdown("Real-time vehicle telemetry monitoring and anomaly detection")

# Auto-update logic
if st.session_state.auto_update:
    current_time = time.time()
    
    # Initialize last_update_time if not set
    if "last_update_time" not in st.session_state:
        st.session_state.last_update_time = current_time
    
    time_since_last_update = current_time - st.session_state.last_update_time
    
    # Sync detector history with session state history before detection
    # Include latest reading if available to ensure LSTM has most recent data
    if hasattr(st.session_state.detector, 'sync_history'):
        history_to_sync = st.session_state.readings_history.copy()
        if "latest_reading" in st.session_state:
            # Include latest reading in history for better LSTM predictions
            history_to_sync.append(st.session_state.latest_reading)
        st.session_state.detector.sync_history(history_to_sync)
    
    # Always generate a new reading when auto-update is enabled
    # This ensures fresh data is available on each page load
    reading = st.session_state.simulator.generate_reading()
    anomaly = st.session_state.detector.detect_anomaly(reading)
    score = st.session_state.detector.get_anomaly_score(reading)
    
    reading["anomaly"] = anomaly
    reading["anomaly_score"] = score
    
    # ALWAYS update latest_reading so display shows current data
    # This ensures the metrics display updates even if not added to history yet
    st.session_state.latest_reading = reading
    
    # Only add to history if enough time has passed (respects update interval)
    if time_since_last_update >= st.session_state.update_interval:
        st.session_state.readings_history.append(reading)
        
        if anomaly == -1:
            recommendation = analyze_anomaly(reading)
            anomaly_data = {
                "timestamp": reading["timestamp"],
                "reading": reading,
                "recommendation": recommendation
            }
            st.session_state.anomalies_detected.append(anomaly_data)
            
            # Set current issue and show notification
            st.session_state.current_issue = anomaly_data
            st.session_state.show_notification = True
        
        # Keep only last 100 readings for performance
        if len(st.session_state.readings_history) > 100:
            st.session_state.readings_history = st.session_state.readings_history[-100:]
        
        st.session_state.last_update_time = current_time
        
        # Sync detector history after adding to session state
        # This ensures detector has the latest readings for next prediction
        if hasattr(st.session_state.detector, 'sync_history'):
            st.session_state.detector.sync_history(st.session_state.readings_history)
    
    # Calculate time until next update
    time_until_next = st.session_state.update_interval - time_since_last_update
    
    # Show refresh status
    st.info(f"⏱️ Auto-updating every {st.session_state.update_interval}s | Next update in {int(max(0, time_until_next))}s")
    
    # Use Streamlit's auto-rerun mechanism
    # Sleep for a short interval then rerun to update data
    import time as time_module
    
    if time_until_next <= 0:
        # Time to update - rerun immediately to refresh data
        st.rerun()
    else:
        # Wait for 1 second then rerun to check again
        # This creates a polling effect that keeps data fresh
        time_module.sleep(1.0)
        st.rerun()

# Display latest anomaly alert with notification banner
if st.session_state.anomalies_detected:
    latest_anomaly = st.session_state.anomalies_detected[-1]
    
    # Show prominent notification banner
    st.markdown("---")
    col1, col2 = st.columns([4, 1])
    with col1:
        st.error("🚨 **ANOMALY DETECTED** - Vehicle issue identified by predictive analysis")
    with col2:
        if st.button("View Details", type="primary"):
            st.session_state.current_issue = latest_anomaly
            st.session_state.current_page = "issue_details"
            st.rerun()
    
    st.markdown("### Latest Anomaly Alert")
    st.markdown(latest_anomaly["recommendation"])
    st.markdown("---")

# Current reading display
# Use latest reading if available (from auto-update), otherwise use last from history
if "latest_reading" in st.session_state:
    latest = st.session_state.latest_reading
elif st.session_state.readings_history:
    latest = st.session_state.readings_history[-1]
else:
    latest = None

if latest:
    
    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Engine RPM",
            f"{latest['sensors']['engine_rpm']:.0f}",
            delta=None
        )
    
    with col2:
        temp_status = "CRITICAL" if latest['sensors']['engine_temp_c'] > 120 else "NORMAL"
        st.metric(
            "Engine Temp",
            f"{latest['sensors']['engine_temp_c']:.1f}°C",
            delta=None
        )
        st.caption(temp_status)
    
    with col3:
        vib_status = "CRITICAL" if latest['sensors']['vibration_level_g'] > 1.0 else "NORMAL"
        st.metric(
            "Vibration",
            f"{latest['sensors']['vibration_level_g']:.3f}g",
            delta=None
        )
        st.caption(vib_status)
    
    with col4:
        st.metric(
            "Throttle",
            f"{latest['sensors']['throttle_pos_pct']}%",
            delta=None
        )
    
    with col5:
        st.metric(
            "Battery",
            f"{latest['sensors']['battery_voltage_v']:.2f}V",
            delta=None
        )
    
    # Anomaly status
    anomaly_status = "ANOMALY DETECTED" if latest['anomaly'] == -1 else "NORMAL"
    st.markdown(f"**Status:** {anomaly_status} | **Anomaly Score:** {latest['anomaly_score']:.3f}")
    
    st.markdown("---")
    
    # Charts
    if len(st.session_state.readings_history) > 1:
        df = pd.DataFrame([
            {
                "timestamp": r["timestamp"],
                "engine_rpm": r["sensors"]["engine_rpm"],
                "engine_temp_c": r["sensors"]["engine_temp_c"],
                "vibration_level_g": r["sensors"]["vibration_level_g"],
                "throttle_pos_pct": r["sensors"]["throttle_pos_pct"],
                "battery_voltage_v": r["sensors"]["battery_voltage_v"],
                "anomaly": r["anomaly"],
                "anomaly_score": r["anomaly_score"]
            }
            for r in st.session_state.readings_history
        ])
        
        # Convert timestamp to datetime for plotting
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Create subplots with better spacing
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=("Engine RPM", "Engine Temperature", "Vibration Level", "Throttle Position", "Battery Voltage", "Anomaly Score"),
            vertical_spacing=0.15,
            horizontal_spacing=0.12,
            row_heights=[0.33, 0.33, 0.34]
        )
        
        # Engine RPM
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["engine_rpm"],
                mode="lines+markers",
                name="RPM",
                line=dict(color="blue", width=2)
            ),
            row=1, col=1
        )
        fig.add_hline(y=3000, line_dash="dash", line_color="orange", row=1, col=1, annotation_text="Max Normal")
        
        # Engine Temperature
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["engine_temp_c"],
                mode="lines+markers",
                name="Temp (°C)",
                line=dict(color="red", width=2)
            ),
            row=1, col=2
        )
        fig.add_hline(y=105, line_dash="dash", line_color="orange", row=1, col=2, annotation_text="Max Normal")
        fig.add_hline(y=120, line_dash="dash", line_color="red", row=1, col=2, annotation_text="Critical")
        
        # Vibration
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["vibration_level_g"],
                mode="lines+markers",
                name="Vibration (g)",
                line=dict(color="purple", width=2)
            ),
            row=2, col=1
        )
        fig.add_hline(y=0.4, line_dash="dash", line_color="orange", row=2, col=1, annotation_text="Max Normal")
        fig.add_hline(y=1.0, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Critical")
        
        # Throttle
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["throttle_pos_pct"],
                mode="lines+markers",
                name="Throttle (%)",
                line=dict(color="green", width=2)
            ),
            row=2, col=2
        )
        
        # Battery Voltage
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["battery_voltage_v"],
                mode="lines+markers",
                name="Battery (V)",
                line=dict(color="orange", width=2)
            ),
            row=3, col=1
        )
        fig.add_hline(y=13.5, line_dash="dash", line_color="green", row=3, col=1, annotation_text="Min Normal")
        fig.add_hline(y=14.5, line_dash="dash", line_color="green", row=3, col=1, annotation_text="Max Normal")
        
        # Anomaly Score (color-coded by anomaly status)
        colors = ["red" if a == -1 else "green" for a in df["anomaly"]]
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["anomaly_score"],
                mode="lines+markers",
                name="Anomaly Score",
                line=dict(color="gray", width=2),
                marker=dict(color=colors, size=8)
            ),
            row=3, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=3, col=2, annotation_text="Anomaly Threshold")
        
        # Update layout with better spacing
        fig.update_layout(
            height=1000,
            showlegend=False,
            title_text="Vehicle Telemetry Dashboard",
            title_x=0.5,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Update x-axis labels
        for i in range(1, 4):
            for j in range(1, 3):
                fig.update_xaxes(title_text="Time", row=i, col=j)
        
        # Update y-axis labels
        fig.update_yaxes(title_text="RPM", row=1, col=1)
        fig.update_yaxes(title_text="°C", row=1, col=2)
        fig.update_yaxes(title_text="g", row=2, col=1)
        fig.update_yaxes(title_text="%", row=2, col=2)
        fig.update_yaxes(title_text="V", row=3, col=1)
        fig.update_yaxes(title_text="Score", row=3, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Anomalies table
        if st.session_state.anomalies_detected:
            st.markdown("### Anomaly History")
            anomalies_df = pd.DataFrame([
                {
                    "Timestamp": a["timestamp"],
                    "RPM": a["reading"]["sensors"]["engine_rpm"],
                    "Temp (°C)": a["reading"]["sensors"]["engine_temp_c"],
                    "Vibration (g)": a["reading"]["sensors"]["vibration_level_g"],
                    "Anomaly Score": a["reading"]["anomaly_score"]
                }
                for a in st.session_state.anomalies_detected[-10:]  # Last 10 anomalies
            ])
            st.dataframe(anomalies_df, use_container_width=True)
else:
    st.info("Click 'Generate New Reading' or enable 'Auto Update' to start monitoring")

