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

# Train model on startup
if not st.session_state.model_trained:
    with st.spinner("Training anomaly detection model on normal vehicle data..."):
        st.session_state.detector.train_initial_model(n_samples=500)
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
        ["None", "Overheat", "Vibration"],
        index=0
    )
    
    if fault_type == "Overheat":
        st.session_state.simulator.inject_fault("overheat")
        st.warning("Overheating fault active")
    elif fault_type == "Vibration":
        st.session_state.simulator.inject_fault("vibration")
        st.warning("Vibration fault active")
    else:
        st.session_state.simulator.inject_fault(None)
        st.success("Normal operation")
    
    st.markdown("---")
    
    st.subheader("Dashboard Controls")
    auto_update = st.checkbox("Auto Update (1s)", value=st.session_state.auto_update)
    st.session_state.auto_update = auto_update
    
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

# Auto-update loop
if st.session_state.auto_update:
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
    
    # Keep only last 100 readings for performance
    if len(st.session_state.readings_history) > 100:
        st.session_state.readings_history = st.session_state.readings_history[-100:]
    
    time.sleep(1)
    st.rerun()

# Display latest anomaly alert
if st.session_state.anomalies_detected:
    latest_anomaly = st.session_state.anomalies_detected[-1]
    st.markdown("---")
    st.markdown("### Latest Anomaly Alert")
    st.markdown(latest_anomaly["recommendation"])
    st.markdown("---")

# Current reading display
if st.session_state.readings_history:
    latest = st.session_state.readings_history[-1]
    
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

