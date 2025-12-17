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
from maintenance_agent import analyze_anomaly

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

# Train model on startup
if not st.session_state.model_trained:
    with st.spinner("Training anomaly detection model on normal vehicle data..."):
        st.session_state.detector.train_initial_model(n_samples=1000)
        st.session_state.model_trained = True
        st.success("Model trained successfully!")

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
        reading = st.session_state.simulator.generate_reading()
        anomaly = st.session_state.detector.detect_anomaly(reading)
        score = st.session_state.detector.get_anomaly_score(reading)
        
        reading["anomaly"] = anomaly
        reading["anomaly_score"] = score
        st.session_state.readings_history.append(reading)
        
        if anomaly == -1:
            recommendation = analyze_anomaly(reading)
            st.session_state.anomalies_detected.append({
                "timestamp": reading["timestamp"],
                "reading": reading,
                "recommendation": recommendation
            })
    
    if st.button("Clear History"):
        st.session_state.readings_history = []
        st.session_state.anomalies_detected = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("**Status:**")
    st.info(f"Readings: {len(st.session_state.readings_history)}")
    st.info(f"Anomalies: {len(st.session_state.anomalies_detected)}")

# Main dashboard
st.title("VehicleCare AI - Predictive Maintenance Dashboard")
st.markdown("Real-time vehicle telemetry monitoring and anomaly detection")

# Auto-update logic
if st.session_state.auto_update:
    current_time = time.time()
    
    # Initialize last_update_time if not set
    if "last_update_time" not in st.session_state:
        st.session_state.last_update_time = current_time
    
    time_since_last_update = current_time - st.session_state.last_update_time
    
    # Always generate a new reading when auto-update is enabled
    # This ensures fresh data is available on each page load
    reading = st.session_state.simulator.generate_reading()
    anomaly = st.session_state.detector.detect_anomaly(reading)
    score = st.session_state.detector.get_anomaly_score(reading)
    
    reading["anomaly"] = anomaly
    reading["anomaly_score"] = score
    
    # Only add to history if enough time has passed (respects update interval)
    if time_since_last_update >= st.session_state.update_interval:
        st.session_state.readings_history.append(reading)
        
        if anomaly == -1:
            recommendation = analyze_anomaly(reading)
            st.session_state.anomalies_detected.append({
                "timestamp": reading["timestamp"],
                "reading": reading,
                "recommendation": recommendation
            })
        
        # Keep only last 100 readings for performance
        if len(st.session_state.readings_history) > 100:
            st.session_state.readings_history = st.session_state.readings_history[-100:]
        
        st.session_state.last_update_time = current_time
    
    # Store the latest reading for display (even if not added to history yet)
    st.session_state.latest_reading = reading
    
    # Calculate time until next update
    time_until_next = max(1, st.session_state.update_interval - time_since_last_update)
    
    # Show refresh status
    st.info(f"⏱️ Auto-updating every {st.session_state.update_interval}s | Next update in {int(time_until_next)}s")
    
    # Use meta refresh tag for reliable auto-refresh in Streamlit
    # This is more reliable than JavaScript in Streamlit's iframe environment
    refresh_delay_seconds = int(time_until_next)
    
    # Inject meta refresh tag
    meta_refresh = f"""
    <meta http-equiv="refresh" content="{refresh_delay_seconds}">
    """
    st.markdown(meta_refresh, unsafe_allow_html=True)
    
    # Also add JavaScript as backup
    refresh_script = f"""
    <script>
        setTimeout(function(){{
            window.location.reload(true);
        }}, {refresh_delay_seconds * 1000});
    </script>
    """
    st.markdown(refresh_script, unsafe_allow_html=True)

# Display latest anomaly alert
if st.session_state.anomalies_detected:
    latest_anomaly = st.session_state.anomalies_detected[-1]
    st.markdown("---")
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

