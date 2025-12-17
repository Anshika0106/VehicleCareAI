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

# Load environment variables from .env file or Streamlit secrets
import os

def load_env_file():
    """Load environment variables from .env file manually."""
    possible_paths = [
        '.env',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'),
    ]
    
    for env_path in possible_paths:
        if os.path.exists(env_path):
            try:
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            if key and value:
                                os.environ[key] = value
                break
            except Exception:
                continue

def load_streamlit_secrets():
    """Load secrets from Streamlit Cloud secrets management."""
    try:
        import streamlit as st
        if hasattr(st, 'secrets'):
            # Copy Streamlit secrets to environment variables
            for key in ['GOOGLE_API_KEY', 'AZURE_SPEECH_KEY', 'AZURE_SPEECH_REGION', 
                       'TWILIO_ACCOUNT_SID', 'TWILIO_AUTH_TOKEN', 'TWILIO_PHONE_NUMBER']:
                if key in st.secrets:
                    os.environ[key] = str(st.secrets[key])
            return True
    except Exception:
        pass
    return False

# Priority: Streamlit secrets > python-dotenv > manual .env loading
if not load_streamlit_secrets():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        load_env_file()

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

# Voice booking imports
import asyncio
import os
from voice_booking_agent import (
    BookingRequest,
    BookingResult,
    BookingStatus,
    AutoBookingProgress,
    book_appointment_automatically,
    auto_book_with_service_centers,
    run_auto_booking_sync,
    SERVICE_CENTER_DIRECTORY,
    get_service_center_phone
)

# Page configuration
st.set_page_config(
    page_title="VehicleCare AI",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# GLOBAL PREMIUM UI STYLING
# ============================================
st.markdown("""
<style>
    /* Import Premium Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
    
    /* Root Variables - Premium Dark Theme */
    :root {
        --bg-primary: #0a0a0b;
        --bg-secondary: #111113;
        --bg-tertiary: #18181b;
        --bg-card: #1c1c1f;
        --bg-card-hover: #222225;
        --border-subtle: #27272a;
        --border-medium: #3f3f46;
        --text-primary: #fafafa;
        --text-secondary: #a1a1aa;
        --text-muted: #71717a;
        --accent-primary: #10b981;
        --accent-secondary: #06b6d4;
        --accent-warning: #f59e0b;
        --accent-danger: #ef4444;
        --accent-gradient: linear-gradient(135deg, #10b981 0%, #06b6d4 100%);
        --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.4);
        --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.5);
        --shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.6);
        --radius-sm: 6px;
        --radius-md: 10px;
        --radius-lg: 16px;
        --radius-xl: 24px;
    }
    
    /* Global Styles */
    .stApp {
        background: var(--bg-primary) !important;
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--border-medium);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-muted);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
    }
    
    /* Main Content Area */
    .main .block-container {
        padding: 2rem 3rem !important;
        max-width: 1400px !important;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }
    
    h1 { font-size: 2.5rem !important; font-weight: 700 !important; }
    h2 { font-size: 1.75rem !important; }
    h3 { font-size: 1.25rem !important; }
    
    p, span, div {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Metric Cards */
    [data-testid="stMetric"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-lg) !important;
        padding: 1.25rem !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stMetric"]:hover {
        background: var(--bg-card-hover) !important;
        border-color: var(--border-medium) !important;
        transform: translateY(-2px);
    }
    
    [data-testid="stMetric"] label {
        color: var(--text-secondary) !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1.75rem !important;
        font-weight: 600 !important;
    }
    
    /* Buttons */
    .stButton > button {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: var(--radius-md) !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.01em !important;
    }
    
    .stButton > button[kind="primary"] {
        background: var(--accent-gradient) !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3) !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4) !important;
    }
    
    .stButton > button[kind="secondary"] {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-medium) !important;
        color: var(--text-primary) !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: var(--bg-card-hover) !important;
        border-color: var(--text-muted) !important;
    }
    
    /* Select boxes and Inputs */
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
        font-family: 'Outfit', sans-serif !important;
    }
    
    .stSelectbox > div > div:hover,
    .stTextInput > div > div > input:hover {
        border-color: var(--border-medium) !important;
    }
    
    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div > input:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2) !important;
    }
    
    /* Labels */
    .stSelectbox label,
    .stTextInput label,
    .stNumberInput label,
    .stCheckbox label {
        color: var(--text-secondary) !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
    }
    
    /* Info/Warning/Error boxes */
    .stAlert {
        border-radius: var(--radius-md) !important;
        border: none !important;
    }
    
    [data-testid="stAlert"][data-baseweb="notification"] {
        background: var(--bg-card) !important;
        border-left: 4px solid var(--accent-primary) !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 500 !important;
    }
    
    .streamlit-expanderContent {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-top: none !important;
        border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        border-radius: var(--radius-lg) !important;
        overflow: hidden !important;
    }
    
    [data-testid="stDataFrame"] > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-lg) !important;
    }
    
    /* Plotly Charts - Dark Theme */
    .js-plotly-plot .plotly .modebar {
        background: var(--bg-card) !important;
    }
    
    /* Checkbox */
    .stCheckbox > label > div[data-testid="stCheckbox"] {
        background: var(--bg-tertiary) !important;
        border-color: var(--border-medium) !important;
    }
    
    /* Caption text */
    .stCaption {
        color: var(--text-muted) !important;
        font-size: 0.8rem !important;
    }
    
    /* Dividers */
    hr {
        border-color: var(--border-subtle) !important;
    }
    
    /* Premium Brand Header */
    .premium-header {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 24px 0;
        margin-bottom: 32px;
        border-bottom: 1px solid var(--border-subtle);
    }
    
    .premium-logo {
        width: 48px;
        height: 48px;
        background: var(--accent-gradient);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
    }
    
    .premium-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }
    
    .premium-subtitle {
        font-family: 'Outfit', sans-serif;
        font-size: 0.875rem;
        color: var(--text-muted);
        margin-top: 2px;
    }
    
    .premium-badge {
        background: var(--accent-gradient);
        color: white;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Status Indicator */
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .status-dot.active {
        background: var(--accent-primary);
        box-shadow: 0 0 8px var(--accent-primary);
    }
    
    .status-dot.warning {
        background: var(--accent-warning);
        box-shadow: 0 0 8px var(--accent-warning);
    }
    
    .status-dot.danger {
        background: var(--accent-danger);
        box-shadow: 0 0 8px var(--accent-danger);
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Premium Cards */
    .premium-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 24px;
        margin: 16px 0;
        transition: all 0.2s ease;
    }
    
    .premium-card:hover {
        background: var(--bg-card-hover);
        border-color: var(--border-medium);
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    
    .premium-card-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 16px;
        padding-bottom: 16px;
        border-bottom: 1px solid var(--border-subtle);
    }
    
    .premium-card-icon {
        width: 44px;
        height: 44px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        background: var(--bg-tertiary);
    }
    
    .premium-card-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .premium-card-subtitle {
        font-family: 'Outfit', sans-serif;
        font-size: 0.8rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .premium-card-content {
        color: var(--text-secondary);
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Severity Badges */
    .severity-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 0.875rem;
        font-weight: 600;
        font-family: 'Outfit', sans-serif;
    }
    
    .severity-critical {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .severity-high {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .severity-medium {
        background: rgba(6, 182, 212, 0.15);
        color: #22d3ee;
        border: 1px solid rgba(6, 182, 212, 0.3);
    }
    
    .severity-low {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    /* Monospace Values */
    .mono-value {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 500;
        color: var(--text-primary);
    }
    
    /* Animated Gradient Text */
    .gradient-text {
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
</style>
""", unsafe_allow_html=True)

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
if "auto_booking_status" not in st.session_state:
    st.session_state.auto_booking_status = None
if "auto_booking_result" not in st.session_state:
    st.session_state.auto_booking_result = None
if "auto_booking_logs" not in st.session_state:
    st.session_state.auto_booking_logs = []
if "booking_in_progress" not in st.session_state:
    st.session_state.booking_in_progress = False
if "auto_booking_triggered" not in st.session_state:
    st.session_state.auto_booking_triggered = False
if "auto_booking_complete" not in st.session_state:
    st.session_state.auto_booking_complete = False
if "calling_centers_progress" not in st.session_state:
    st.session_state.calling_centers_progress = []

# Default customer info (would come from user profile in production)
if "customer_info" not in st.session_state:
    st.session_state.customer_info = {
        "name": "John Doe",
        "phone": "+1 (555) 123-4567",
        "email": "john.doe@example.com"
    }

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
    """Render the Issue Detected page with premium dark design."""
    
    # Get issue data first
    if st.session_state.current_issue:
        issue = st.session_state.current_issue
        reading = issue["reading"]
        issue_title, issue_description, recommended_action = get_issue_details(reading)
        severity = get_severity_level(reading)
        
        # Determine severity styling for dark theme
        severity_styles = {
            "Critical": {"class": "severity-critical", "icon": "●", "glow": "#ef4444"},
            "High": {"class": "severity-high", "icon": "●", "glow": "#f59e0b"},
            "Medium": {"class": "severity-medium", "icon": "●", "glow": "#06b6d4"},
            "Low": {"class": "severity-low", "icon": "●", "glow": "#10b981"}
        }
        sev_style = severity_styles.get(severity, severity_styles["Medium"])
    else:
        issue_title = "No Issue"
        issue_description = "No issue data available."
        recommended_action = "Return to dashboard."
        severity = "Unknown"
        sev_style = {"class": "severity-medium", "icon": "○", "glow": "#71717a"}
    
    # Premium Dark Theme Styles
    st.markdown(
        f"""
        <style>
        .issue-container {{
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .issue-header {{
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 32px;
            padding-bottom: 24px;
            border-bottom: 1px solid #27272a;
        }}
        
        .issue-logo-icon {{
            width: 56px;
            height: 56px;
            background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%);
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            box-shadow: 0 8px 24px rgba(16, 185, 129, 0.3);
        }}
        
        .issue-header-text {{
            flex: 1;
        }}
        
        .issue-header-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.5rem;
            font-weight: 700;
            color: #fafafa;
            letter-spacing: -0.02em;
        }}
        
        .issue-header-subtitle {{
            font-family: 'Outfit', sans-serif;
            font-size: 0.875rem;
            color: #71717a;
            margin-top: 4px;
        }}
        
        .alert-banner {{
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%);
            border: 1px solid rgba(239, 68, 68, 0.2);
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .alert-banner-icon {{
            font-size: 24px;
            animation: pulse-alert 2s ease-in-out infinite;
        }}
        
        @keyframes pulse-alert {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.7; transform: scale(1.1); }}
        }}
        
        .alert-banner-text {{
            font-family: 'Outfit', sans-serif;
            color: #fca5a5;
            font-size: 0.95rem;
            font-weight: 500;
        }}
        
        .issue-card {{
            background: #1c1c1f;
            border: 1px solid #27272a;
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 16px;
            transition: all 0.2s ease;
        }}
        
        .issue-card:hover {{
            background: #222225;
            border-color: #3f3f46;
            transform: translateY(-2px);
        }}
        
        .issue-card-header {{
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 16px;
        }}
        
        .issue-card-icon {{
            width: 44px;
            height: 44px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }}
        
        .issue-card-icon.danger {{
            background: rgba(239, 68, 68, 0.15);
            box-shadow: 0 0 20px rgba(239, 68, 68, 0.1);
        }}
        
        .issue-card-icon.info {{
            background: rgba(6, 182, 212, 0.15);
            box-shadow: 0 0 20px rgba(6, 182, 212, 0.1);
        }}
        
        .issue-card-icon.action {{
            background: rgba(16, 185, 129, 0.15);
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.1);
        }}
        
        .issue-card-label {{
            font-family: 'Outfit', sans-serif;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #71717a;
        }}
        
        .issue-card-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.1rem;
            font-weight: 600;
            color: #fafafa;
            margin-top: 2px;
        }}
        
        .issue-card-content {{
            font-family: 'Outfit', sans-serif;
            color: #a1a1aa;
            font-size: 0.95rem;
            line-height: 1.7;
        }}
        
        .severity-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
        }}
        
        .severity-label {{
            font-family: 'Outfit', sans-serif;
            color: #71717a;
            font-size: 0.9rem;
        }}
        
        .action-box {{
            background: rgba(16, 185, 129, 0.08);
            border-left: 3px solid #10b981;
            padding: 16px 20px;
            border-radius: 0 10px 10px 0;
            margin-top: 8px;
        }}
        
        .action-box p {{
            font-family: 'Outfit', sans-serif;
            color: #34d399;
            font-size: 0.95rem;
            line-height: 1.6;
            margin: 0;
        }}
        
        .booking-status {{
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(6, 182, 212, 0.1) 100%);
            border: 1px solid rgba(16, 185, 129, 0.2);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            margin-top: 24px;
        }}
        
        .booking-status-icon {{
            font-size: 32px;
            margin-bottom: 12px;
        }}
        
        .booking-status-text {{
            font-family: 'Outfit', sans-serif;
            color: #34d399;
            font-size: 1rem;
            font-weight: 500;
        }}
        
        .booking-status-subtext {{
            font-family: 'Outfit', sans-serif;
            color: #71717a;
            font-size: 0.85rem;
            margin-top: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Back button with premium styling
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        if st.button("← Back", type="secondary"):
            st.session_state.current_page = "dashboard"
            st.session_state.show_notification = False
            st.rerun()
    
    st.markdown('<div class="issue-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown('''
        <div class="issue-header">
            <div class="issue-logo-icon" style="font-size: 18px; font-weight: 700; color: white;">VC</div>
            <div class="issue-header-text">
                <div class="issue-header-title">VehicleCare AI</div>
                <div class="issue-header-subtitle">Predictive Maintenance Alert</div>
            </div>
        </div>
    ''', unsafe_allow_html=True)
    
    if st.session_state.current_issue:
        # Alert Banner
        st.markdown(f'''
            <div class="alert-banner">
                <span class="alert-banner-icon" style="color: #fca5a5; font-weight: bold;">!</span>
                <span class="alert-banner-text">Anomaly detected - Automated service booking initiated</span>
            </div>
        ''', unsafe_allow_html=True)
        
        # Issue Detected Card
        st.markdown(f'''
            <div class="issue-card">
                <div class="issue-card-header">
                    <div class="issue-card-icon danger" style="font-weight: bold; color: #f87171;">!</div>
                    <div>
                        <div class="issue-card-label">Issue Identified</div>
                        <div class="issue-card-title">{issue_title}</div>
                    </div>
                </div>
                <div class="issue-card-content">{issue_description}</div>
            </div>
        ''', unsafe_allow_html=True)
        
        # Severity Card
        st.markdown(f'''
            <div class="issue-card">
                <div class="issue-card-header">
                    <div class="issue-card-icon info" style="font-weight: bold; color: #22d3ee;">i</div>
                    <div>
                        <div class="issue-card-label">Diagnostic Analysis</div>
                        <div class="issue-card-title">Severity Assessment</div>
                    </div>
                </div>
                <div class="severity-row">
                    <span class="severity-label">Risk Level</span>
                    <span class="{sev_style['class']}">{sev_style['icon']} {severity}</span>
                </div>
            </div>
        ''', unsafe_allow_html=True)
        
        # Action Card with Auto-booking status
        st.markdown(f'''
            <div class="issue-card">
                <div class="issue-card-header">
                    <div class="issue-card-icon action" style="font-weight: bold; color: #34d399;">AI</div>
                    <div>
                        <div class="issue-card-label">Automated Response</div>
                        <div class="issue-card-title">AI Service Booking</div>
                    </div>
                </div>
                <div class="action-box">
                    <p>VehicleCare AI is automatically contacting service centers to schedule your appointment. No action required.</p>
                </div>
            </div>
            
            <div class="booking-status">
                <div class="booking-status-icon" style="font-size: 24px; color: #34d399;">●</div>
                <div class="booking-status-text">Initiating automated booking...</div>
                <div class="booking-status-subtext">Calling service centers to find the best available slot</div>
            </div>
        ''', unsafe_allow_html=True)
        
        # Auto-trigger booking
        if not st.session_state.auto_booking_triggered:
            st.session_state.auto_booking_triggered = True
            st.session_state.current_page = "auto_booking_progress"
            time.sleep(1)  # Brief pause for UX
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_auto_booking_progress_page():
    """Render the Auto-Booking Progress page with premium dark design."""
    
    st.markdown(
        """
        <style>
        .booking-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .booking-header {
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 24px;
            border-bottom: 1px solid #27272a;
        }
        
        .booking-logo-row {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            margin-bottom: 20px;
        }
        
        .booking-logo-icon {
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            box-shadow: 0 8px 24px rgba(16, 185, 129, 0.3);
        }
        
        .booking-logo-text {
            font-family: 'Outfit', sans-serif;
            font-size: 1.5rem;
            font-weight: 700;
            color: #fafafa;
        }
        
        .booking-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.5rem;
            font-weight: 600;
            color: #fafafa;
            margin-bottom: 8px;
        }
        
        .booking-subtitle {
            font-family: 'Outfit', sans-serif;
            color: #71717a;
            font-size: 0.95rem;
        }
        
        .progress-indicator {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            margin: 24px 0;
            padding: 16px;
            background: rgba(16, 185, 129, 0.08);
            border: 1px solid rgba(16, 185, 129, 0.2);
            border-radius: 12px;
        }
        
        .progress-spinner {
            width: 24px;
            height: 24px;
            border: 3px solid #27272a;
            border-top-color: #10b981;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .progress-text {
            font-family: 'Outfit', sans-serif;
            color: #34d399;
            font-size: 0.95rem;
            font-weight: 500;
        }
        
        .center-card {
            background: #1c1c1f;
            border: 1px solid #27272a;
            border-radius: 12px;
            padding: 18px 20px;
            margin: 10px 0;
            display: flex;
            align-items: center;
            gap: 16px;
            transition: all 0.3s ease;
        }
        
        .center-card.calling {
            border-color: #06b6d4;
            background: rgba(6, 182, 212, 0.08);
            animation: glow-pulse 2s ease-in-out infinite;
        }
        
        @keyframes glow-pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(6, 182, 212, 0); }
            50% { box-shadow: 0 0 20px 0 rgba(6, 182, 212, 0.3); }
        }
        
        .center-card.success {
            border-color: #10b981;
            background: rgba(16, 185, 129, 0.1);
            box-shadow: 0 0 30px rgba(16, 185, 129, 0.2);
        }
        
        .center-card.failed {
            border-color: #3f3f46;
            background: #18181b;
            opacity: 0.6;
        }
        
        .center-card.waiting {
            opacity: 0.4;
        }
        
        .center-icon {
            font-size: 24px;
            width: 44px;
            height: 44px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 10px;
            background: #27272a;
        }
        
        .center-icon.calling {
            background: rgba(6, 182, 212, 0.15);
            animation: phone-ring 1s ease-in-out infinite;
        }
        
        @keyframes phone-ring {
            0%, 100% { transform: rotate(0deg); }
            25% { transform: rotate(15deg); }
            75% { transform: rotate(-15deg); }
        }
        
        .center-icon.success {
            background: rgba(16, 185, 129, 0.15);
        }
        
        .center-info {
            flex: 1;
        }
        
        .center-name {
            font-family: 'Outfit', sans-serif;
            font-size: 1rem;
            font-weight: 600;
            color: #fafafa;
        }
        
        .center-status {
            font-family: 'Outfit', sans-serif;
            font-size: 0.8rem;
            color: #71717a;
            margin-top: 4px;
        }
        
        .center-status.calling {
            color: #22d3ee;
            font-weight: 500;
        }
        
        .center-status.success {
            color: #34d399;
            font-weight: 600;
        }
        
        .center-status.failed {
            color: #ef4444;
        }
        
        .call-animation {
            display: inline-block;
            animation: ring 1s ease-in-out infinite;
        }
        
        @keyframes ring {
            0%, 100% { transform: rotate(0deg); }
            25% { transform: rotate(15deg); }
            75% { transform: rotate(-15deg); }
        }
        .center-status.failed {
            color: #71717a;
        }
        
        .success-banner {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 182, 212, 0.1) 100%);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            margin-top: 24px;
        }
        
        .success-icon {
            font-size: 48px;
            margin-bottom: 16px;
        }
        
        .success-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.25rem;
            font-weight: 700;
            color: #34d399;
            margin-bottom: 8px;
        }
        
        .success-subtitle {
            font-family: 'Outfit', sans-serif;
            color: #a1a1aa;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="booking-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown('''
        <div class="booking-header">
            <div class="booking-logo-row">
                <div class="booking-logo-icon" style="font-size: 16px; font-weight: 700; color: white;">VC</div>
                <span class="booking-logo-text">VehicleCare AI</span>
            </div>
            <div class="booking-title">Automated Booking in Progress</div>
            <div class="booking-subtitle">AI is calling service centers to find the best available appointment</div>
        </div>
    ''', unsafe_allow_html=True)
    
    # Get issue info
    if st.session_state.current_issue:
        issue = st.session_state.current_issue
        reading = issue["reading"]
        issue_title, issue_description, _ = get_issue_details(reading)
        severity = get_severity_level(reading)
        
        # Show issue summary
        st.info(f"**Issue:** {issue_title} | **Severity:** {severity}")
        
        # Initialize progress tracking
        service_centers = list(SERVICE_CENTER_DIRECTORY.keys())
        
        # Progress placeholder
        progress_container = st.container()
        status_placeholder = st.empty()
        
        # Run the auto-booking process
        if not st.session_state.auto_booking_complete:
            
            # Show initial state - all centers waiting
            with progress_container:
                for idx, center in enumerate(service_centers):
                    st.markdown(f'''
                        <div class="center-card waiting">
                            <div class="center-icon" style="font-weight: 600; color: #71717a;">SC</div>
                            <div class="center-info">
                                <div class="center-name">{center}</div>
                                <div class="center-status">Waiting...</div>
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)
            
            # Run the booking
            progress_updates = []
            
            def progress_callback(progress: AutoBookingProgress):
                progress_updates.append(progress)
            
            # Execute auto-booking
            try:
                customer = st.session_state.customer_info
                result = run_auto_booking_sync(
                    customer_name=customer["name"],
                    customer_phone=customer["phone"],
                    customer_email=customer["email"],
                    vehicle_id=st.session_state.simulator.vehicle_id,
                    issue_type=issue_title,
                    issue_description=issue_description,
                    severity=severity,
                    google_api_key=os.getenv("GOOGLE_API_KEY", "demo-key"),
                    azure_speech_key=os.getenv("AZURE_SPEECH_KEY", ""),
                    azure_speech_region=os.getenv("AZURE_SPEECH_REGION", "eastus"),
                    progress_callback=progress_callback
                )
                
                st.session_state.auto_booking_result = result
                st.session_state.auto_booking_complete = True
                st.session_state.calling_centers_progress = progress_updates
                
                if result.status == BookingStatus.CONFIRMED:
                    # Create appointment
                    appointment = {
                        "service_center": result.service_center,
                        "service_type": issue_title,
                        "date": datetime.now().date() + timedelta(days=1),
                        "time": result.scheduled_time or "10:00 AM",
                        "customer_name": customer["name"],
                        "customer_phone": customer["phone"],
                        "customer_email": customer["email"],
                        "issue": issue_title,
                        "status": "Confirmed (Auto-Booked)",
                        "confirmation_number": result.confirmation_number,
                        "booking_method": "Automated AI Call",
                        "call_transcript": result.call_transcript,
                        "created_at": datetime.now()
                    }
                    
                    st.session_state.appointments.append(appointment)
                    st.session_state.latest_appointment = appointment
                    st.session_state.current_page = "confirmation"
                    st.rerun()
                else:
                    status_placeholder.error(f"Could not book with any service center: {result.notes}")
                    
            except Exception as e:
                st.session_state.auto_booking_complete = True
                status_placeholder.error(f"Booking failed: {str(e)}")
        
        # Show final progress - only show final status per center (not duplicates)
        if st.session_state.calling_centers_progress:
            progress_container.empty()
            
            # Get only the final status for each center (skip "calling" if there's a result)
            final_status_per_center = {}
            for progress in st.session_state.calling_centers_progress:
                center = progress.current_center
                # Always update - later statuses override earlier ones
                final_status_per_center[center] = progress
            
            with progress_container:
                for center_name in service_centers:
                    if center_name in final_status_per_center:
                        progress = final_status_per_center[center_name]
                        if progress.status == "confirmed":
                            card_class = "success"
                            icon = "✓"
                            status_text = "Booking Confirmed!"
                        elif progress.status == "calling":
                            card_class = "calling"
                            icon = "●"
                            status_text = "Calling..."
                        else:
                            card_class = "failed"
                            icon = "×"
                            status_text = "No availability"
                    else:
                        # Center not yet called
                        card_class = "waiting"
                        icon = "🏢"
                        status_text = "Waiting..."
                    
                    st.markdown(f'''
                        <div class="center-card {card_class}">
                            <div class="center-icon">{icon}</div>
                            <div class="center-info">
                                <div class="center-name">{center_name}</div>
                                <div class="center-status {card_class}">{status_text}</div>
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)
        
        # Retry button if failed
        if st.session_state.auto_booking_complete and (
            not st.session_state.auto_booking_result or 
            st.session_state.auto_booking_result.status != BookingStatus.CONFIRMED
        ):
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Retry Booking", type="primary", use_container_width=True):
                    st.session_state.auto_booking_complete = False
                    st.session_state.calling_centers_progress = []
                    st.rerun()
            with col2:
                if st.button("← Back to Dashboard", type="secondary", use_container_width=True):
                    st.session_state.current_page = "dashboard"
                    st.session_state.auto_booking_triggered = False
                    st.session_state.auto_booking_complete = False
                    st.session_state.show_notification = False
                    st.rerun()
    else:
        st.warning("No issue detected. Returning to dashboard...")
        st.session_state.current_page = "dashboard"
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_schedule_service_page():
    """Render the Schedule Service page with automated booking option."""
    
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
        
        .schedule-header {
            text-align: center;
            margin-bottom: 40px;
        }
        
        .schedule-logo {
            font-family: 'DM Sans', sans-serif;
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(135deg, #1e3a5f 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .booking-mode-card {
            background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 16px;
            padding: 24px;
            margin: 16px 0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.07);
            border: 2px solid #e2e8f0;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .booking-mode-card:hover {
            border-color: #3b82f6;
            transform: translateY(-2px);
            box-shadow: 0 8px 15px -3px rgba(59, 130, 246, 0.15);
        }
        
        .booking-mode-card.selected {
            border-color: #3b82f6;
            background: linear-gradient(145deg, #eff6ff 0%, #dbeafe 100%);
        }
        
        .mode-icon {
            font-size: 32px;
            margin-bottom: 12px;
        }
        
        .mode-title {
            font-family: 'DM Sans', sans-serif;
            font-size: 18px;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 8px;
        }
        
        .mode-description {
            font-family: 'DM Sans', sans-serif;
            font-size: 14px;
            color: #64748b;
            line-height: 1.5;
        }
        
        .ai-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin-left: 8px;
        }
        
        .call-log-container {
            background: #1e293b;
            border-radius: 12px;
            padding: 16px;
            margin: 16px 0;
            max-height: 300px;
            overflow-y: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
        }
        
        .call-log-entry {
            padding: 8px 0;
            border-bottom: 1px solid #334155;
        }
        
        .call-log-entry:last-child {
            border-bottom: none;
        }
        
        .call-log-time {
            color: #64748b;
            font-size: 11px;
        }
        
        .call-log-status {
            color: #22c55e;
        }
        
        .call-log-ai {
            color: #8b5cf6;
        }
        
        .call-log-service {
            color: #3b82f6;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('''
        <div class="schedule-header">
            <div class="schedule-logo">VehicleCare AI</div>
        </div>
    ''', unsafe_allow_html=True)
    
    # Back button
    if st.button("← Back"):
        st.session_state.current_page = "issue_details"
        st.session_state.auto_booking_status = None
        st.session_state.auto_booking_logs = []
        st.rerun()
    
    st.markdown("### 📅 Schedule Service Appointment")
    
    if st.session_state.current_issue:
        issue = st.session_state.current_issue
        reading = issue["reading"]
        issue_title, issue_description, _ = get_issue_details(reading)
        severity = get_severity_level(reading)
        
        # Show issue summary
        st.info(f"**Issue:** {issue_title} | **Severity:** {severity}")
        
        # Booking mode selection
        st.markdown("#### Choose Booking Method")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Auto-Book via AI Call", type="primary", use_container_width=True, 
                        help="Our AI will call the service center and book the appointment for you"):
                st.session_state.booking_mode = "auto"
        
        with col2:
            if st.button("✍️ Manual Booking", type="secondary", use_container_width=True,
                        help="Fill in the form yourself"):
                st.session_state.booking_mode = "manual"
        
        # Initialize booking mode
        if "booking_mode" not in st.session_state:
            st.session_state.booking_mode = None
        
        st.markdown("---")
        
        # Service center selection (common to both modes)
        service_centers = list(SERVICE_CENTER_DIRECTORY.keys())
        selected_center = st.selectbox("Select Service Center", service_centers)
        
        # Show service center info
        center_info = SERVICE_CENTER_DIRECTORY.get(selected_center, {})
        if center_info:
            st.caption(f"📍 {center_info.get('address', '')} | ⏰ {center_info.get('hours', '')}")
        
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
        
        service_options = sorted(list(set(service_types.values()))) + ["General Inspection & Diagnosis"]
        default_service = service_types.get(issue_title, "General Inspection & Diagnosis")
        
        try:
            default_index = service_options.index(default_service)
        except ValueError:
            default_index = len(service_options) - 1
        
        service_type = st.selectbox("Service Type", service_options, index=default_index)
        
        # Date and time selection
        col1, col2 = st.columns(2)
        with col1:
            min_date = datetime.now().date() + timedelta(days=1)
            max_date = datetime.now().date() + timedelta(days=30)
            selected_date = st.date_input("Preferred Date", min_value=min_date, max_value=max_date, value=min_date)
        
        with col2:
            time_slots = [
                "08:00 AM", "09:00 AM", "10:00 AM", "11:00 AM",
                "12:00 PM", "01:00 PM", "02:00 PM", "03:00 PM",
                "04:00 PM", "05:00 PM"
            ]
            selected_time = st.selectbox("Preferred Time", time_slots)
        
        # Customer information
        st.markdown("#### Customer Information")
        col1, col2 = st.columns(2)
        with col1:
            customer_name = st.text_input("Name", value="John Doe")
            customer_phone = st.text_input("Phone", value="+1 (555) 123-4567")
        with col2:
            customer_email = st.text_input("Email", value="john.doe@example.com")
            vehicle_id = st.text_input("Vehicle ID", value=st.session_state.simulator.vehicle_id)
        
        st.markdown("---")
        
        # AUTO-BOOKING MODE
        if st.session_state.booking_mode == "auto":
            st.markdown("### AI-Powered Automated Booking")
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                        border-left: 4px solid #0284c7; padding: 16px 20px; 
                        border-radius: 0 12px 12px 0; margin-bottom: 20px;">
                <p style="margin: 0; color: #0c4a6e; font-size: 14px; line-height: 1.6;">
                    <strong>How it works:</strong><br>
                    1. Our AI agent will call the service center on your behalf<br>
                    2. It will explain your vehicle issue and request an appointment<br>
                    3. The AI will negotiate the best available time slot<br>
                    4. You'll receive confirmation when booking is complete
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # API Configuration check
            api_configured = all([
                os.getenv("GOOGLE_API_KEY"),
                os.getenv("AZURE_SPEECH_KEY"),
                os.getenv("AZURE_SPEECH_REGION")
            ])
            
            if not api_configured:
                st.warning("⚠️ **Demo Mode**: API keys not configured. Running in simulation mode.")
            
            # Start auto-booking button
            if not st.session_state.booking_in_progress:
                if st.button("🚀 Start AI Booking Call", type="primary", use_container_width=True):
                    st.session_state.booking_in_progress = True
                    st.session_state.auto_booking_logs = []
                    st.session_state.auto_booking_status = "initiating"
                    st.rerun()
            
            # Show booking progress
            if st.session_state.booking_in_progress:
                st.markdown("### Call in Progress...")
                
                # Progress indicator
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                log_placeholder = st.empty()
                
                # Create booking request
                booking_request = BookingRequest(
                    customer_name=customer_name,
                    customer_phone=customer_phone,
                    customer_email=customer_email,
                    vehicle_id=vehicle_id,
                    issue_type=issue_title,
                    issue_description=issue_description,
                    severity=severity,
                    preferred_date=datetime.combine(selected_date, datetime.min.time()),
                    preferred_time=selected_time,
                    service_center_phone=get_service_center_phone(selected_center),
                    service_center_name=selected_center
                )
                
                # Status callback function
                def status_callback(status: BookingStatus, message: str):
                    st.session_state.auto_booking_logs.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "status": status.value,
                        "message": message
                    })
                
                # Run the booking asynchronously
                async def run_booking():
                    result = await book_appointment_automatically(
                        booking_request=booking_request,
                        google_api_key=os.getenv("GOOGLE_API_KEY", "demo-key"),
                        azure_speech_key=os.getenv("AZURE_SPEECH_KEY", "demo-key"),
                        azure_speech_region=os.getenv("AZURE_SPEECH_REGION", "eastus"),
                        status_callback=status_callback
                    )
                    return result
                
                # Execute booking
                try:
                    result = asyncio.run(run_booking())
                    st.session_state.auto_booking_result = result
                    st.session_state.booking_in_progress = False
                    
                    if result.status == BookingStatus.CONFIRMED:
                        # Create appointment from result
                        appointment = {
                            "service_center": selected_center,
                            "service_type": service_type,
                            "date": selected_date,
                            "time": result.scheduled_time or selected_time,
                            "customer_name": customer_name,
                            "customer_phone": customer_phone,
                            "customer_email": customer_email,
                            "issue": issue_title,
                            "status": "Confirmed (AI Booked)",
                            "confirmation_number": result.confirmation_number,
                            "booking_method": "Automated AI Call",
                            "call_transcript": result.call_transcript,
                            "created_at": datetime.now()
                        }
                        
                        st.session_state.appointments.append(appointment)
                        st.session_state.latest_appointment = appointment
                        st.session_state.current_page = "confirmation"
                        st.rerun()
                    else:
                        st.error(f"Booking failed: {result.notes}")
                        st.session_state.booking_in_progress = False
                        
                except Exception as e:
                    st.error(f"Error during booking: {str(e)}")
                    st.session_state.booking_in_progress = False
                
                # Display call logs
                if st.session_state.auto_booking_logs:
                    st.markdown("#### Call Log")
                    log_html = '<div class="call-log-container">'
                    for log in st.session_state.auto_booking_logs:
                        speaker_class = "call-log-ai" if "AI:" in log["message"] else "call-log-service"
                        if "Service Center:" in log["message"]:
                            speaker_class = "call-log-service"
                        log_html += f'''
                        <div class="call-log-entry">
                            <span class="call-log-time">[{log["time"]}]</span>
                            <span class="call-log-status">[{log["status"]}]</span>
                            <span class="{speaker_class}">{log["message"]}</span>
                        </div>
                        '''
                    log_html += '</div>'
                    st.markdown(log_html, unsafe_allow_html=True)
        
        # MANUAL BOOKING MODE  
        elif st.session_state.booking_mode == "manual":
            st.markdown("### ✍️ Manual Booking")
            
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
                    "booking_method": "Manual",
                    "created_at": datetime.now()
                }
                
                st.session_state.appointments.append(appointment)
                st.session_state.latest_appointment = appointment
                st.session_state.current_page = "confirmation"
                st.rerun()
        
        else:
            # No mode selected yet - show instructions
            st.markdown("""
            <div style="text-align: center; padding: 40px; color: #64748b;">
                <p style="font-size: 16px;">👆 Select a booking method above to continue</p>
                <p style="font-size: 14px;">
                    <strong>Auto-Book</strong>: Our AI calls the service center for you<br>
                    <strong>Manual</strong>: Fill in the form yourself
                </p>
            </div>
            """, unsafe_allow_html=True)


def render_confirmation_page():
    """Render the Appointment Confirmation page with premium dark design."""
    st.markdown(
        """
        <style>
        .confirm-container {
            max-width: 700px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .confirm-header {
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 24px;
            border-bottom: 1px solid #27272a;
        }
        
        .confirm-logo-row {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            margin-bottom: 24px;
        }
        
        .confirm-logo-icon {
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            box-shadow: 0 8px 24px rgba(16, 185, 129, 0.3);
        }
        
        .confirm-logo-text {
            font-family: 'Outfit', sans-serif;
            font-size: 1.5rem;
            font-weight: 700;
            color: #fafafa;
        }
        
        .confirm-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.75rem;
            font-weight: 700;
            color: #fafafa;
            margin-bottom: 8px;
        }
        
        .confirm-subtitle {
            font-family: 'Outfit', sans-serif;
            color: #71717a;
            font-size: 0.95rem;
        }
        
        .success-banner {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(6, 182, 212, 0.15) 100%);
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 24px;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 28px;
            box-shadow: 0 0 40px rgba(16, 185, 129, 0.15);
        }
        
        .success-icon {
            font-size: 56px;
            margin-bottom: 16px;
        }
        
        .success-text {
            font-family: 'Outfit', sans-serif;
            font-size: 1.25rem;
            font-weight: 700;
            color: #34d399;
            letter-spacing: 0.02em;
        }
        
        .details-card {
            background: #1c1c1f;
            border: 1px solid #27272a;
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 16px;
            transition: all 0.2s ease;
        }
        
        .details-card:hover {
            background: #222225;
            border-color: #3f3f46;
        }
        
        .details-card-header {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 20px;
            padding-bottom: 16px;
            border-bottom: 1px solid #27272a;
        }
        
        .details-card-icon {
            width: 44px;
            height: 44px;
            border-radius: 10px;
            background: rgba(6, 182, 212, 0.15);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }
        
        .details-card-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.1rem;
            font-weight: 600;
            color: #fafafa;
        }
        
        .details-row {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding: 12px 0;
            border-bottom: 1px solid #1f1f23;
        }
        
        .details-row:last-child {
            border-bottom: none;
        }
        
        .details-label {
            font-family: 'Outfit', sans-serif;
            color: #71717a;
            font-size: 0.875rem;
            font-weight: 500;
        }
        
        .details-value {
            font-family: 'IBM Plex Mono', monospace;
            color: #fafafa;
            font-size: 0.9rem;
            font-weight: 500;
            text-align: right;
            max-width: 60%;
        }
        
        .info-banner {
            background: rgba(245, 158, 11, 0.1);
            border-left: 3px solid #f59e0b;
            padding: 16px 20px;
            border-radius: 0 10px 10px 0;
            margin-bottom: 28px;
            display: flex;
            align-items: center;
            gap: 14px;
        }
        
        .info-banner-icon {
            font-size: 24px;
        }
        
        .info-banner-text {
            font-family: 'Outfit', sans-serif;
            color: #fbbf24;
            font-size: 0.9rem;
            font-weight: 500;
            line-height: 1.5;
        }
        
        /* Button styling */
        div[data-testid="stButton"] > button {
            font-family: 'DM Sans', sans-serif !important;
            font-weight: 600 !important;
            font-size: 15px !important;
            padding: 14px 24px !important;
            border-radius: 12px !important;
            transition: all 0.3s ease !important;
        }
        
        div[data-testid="stButton"] > button[kind="secondary"] {
            background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%) !important;
            color: #334155 !important;
            border: 2px solid #e2e8f0 !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
        }
        
        div[data-testid="stButton"] > button[kind="secondary"]:hover {
            background: #f1f5f9 !important;
            border-color: #cbd5e1 !important;
            transform: translateY(-1px) !important;
        }
        
        div[data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important;
            color: #ffffff !important;
            border: none !important;
            box-shadow: 0 4px 14px rgba(30, 41, 59, 0.25) !important;
        }
        
        div[data-testid="stButton"] > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #334155 0%, #475569 100%) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 20px rgba(30, 41, 59, 0.3) !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="confirm-container">', unsafe_allow_html=True)
    
    # Header with premium styling
    st.markdown('''
        <div class="confirm-header">
            <div class="confirm-logo-row">
                <div class="confirm-logo-icon" style="font-size: 16px; font-weight: 700; color: white;">VC</div>
                <span class="confirm-logo-text">VehicleCare AI</span>
            </div>
            <div class="confirm-title">Appointment Confirmed</div>
            <div class="confirm-subtitle">Your service appointment has been successfully scheduled</div>
        </div>
    ''', unsafe_allow_html=True)
    
    if st.session_state.latest_appointment:
        appointment = st.session_state.latest_appointment
        
        # Success Banner
        st.markdown('''
            <div class="success-banner">
                <div class="success-icon" style="font-size: 40px; color: #34d399;">✓</div>
                <div class="success-text">Booking Successful</div>
            </div>
        ''', unsafe_allow_html=True)
        
        # Appointment Details Card
        date_str = appointment["date"].strftime("%A, %B %d, %Y")
        confirmation_num = appointment.get("confirmation_number", "N/A")
        booking_method = appointment.get("booking_method", "Manual")
        
        # Show AI badge if auto-booked
        method_badge = ""
        if booking_method == "Automated AI Call":
            method_badge = '<span style="background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%); color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; margin-left: 8px;">AI Booked</span>'
        
        st.markdown(f'''
            <div class="details-card">
                <div class="details-card-header">
                    <div class="details-card-icon" style="font-weight: 600; color: #22d3ee;">≡</div>
                    <div class="details-card-title">Appointment Details {method_badge}</div>
                </div>
                <div class="details-row">
                    <span class="details-label">Confirmation #</span>
                    <span class="details-value">{confirmation_num}</span>
                </div>
                <div class="details-row">
                    <span class="details-label">Service Center</span>
                    <span class="details-value">{appointment["service_center"]}</span>
                </div>
                <div class="details-row">
                    <span class="details-label">Service Type</span>
                    <span class="details-value">{appointment["service_type"]}</span>
                </div>
                <div class="details-row">
                    <span class="details-label">Date</span>
                    <span class="details-value">{date_str}</span>
                </div>
                <div class="details-row">
                    <span class="details-label">Time</span>
                    <span class="details-value">{appointment["time"]}</span>
                </div>
                <div class="details-row">
                    <span class="details-label">Booking Method</span>
                    <span class="details-value">{booking_method}</span>
                </div>
            </div>
        ''', unsafe_allow_html=True)
        
        # Show call transcript if available (for AI bookings)
        if appointment.get("call_transcript"):
            with st.expander("View Call Transcript"):
                st.markdown(f'''
                    <div style="background: #1e293b; color: #e2e8f0; padding: 16px; 
                                border-radius: 8px; font-family: monospace; font-size: 13px;
                                white-space: pre-wrap; max-height: 400px; overflow-y: auto;">
{appointment["call_transcript"]}
                    </div>
                ''', unsafe_allow_html=True)
        
        # Info Banner
        st.markdown('''
            <div class="info-banner">
                <span class="info-banner-icon" style="color: #fbbf24;">i</span>
                <span class="info-banner-text">Please arrive 10 minutes early and bring your vehicle registration documents.</span>
            </div>
        ''', unsafe_allow_html=True)
        
        # Action Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("View Appointments", type="secondary", use_container_width=True):
                st.session_state.current_page = "appointments"
                st.rerun()
        
        with col2:
            if st.button("Back to Dashboard", type="primary", use_container_width=True):
                st.session_state.current_page = "dashboard"
                st.session_state.show_notification = False
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_vehicle_health_dashboard():
    """Render the Vehicle Health Dashboard with premium dark design."""
    st.markdown(
        """
        <style>
        .health-container {
            max-width: 900px;
            margin: 0 auto;
        }
        
        .health-header {
            display: flex;
            align-items: center;
            gap: 16px;
            padding-bottom: 24px;
            margin-bottom: 32px;
            border-bottom: 1px solid #27272a;
        }
        
        .health-logo {
            width: 56px;
            height: 56px;
            background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%);
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            box-shadow: 0 8px 24px rgba(16, 185, 129, 0.3);
        }
        
        .health-header-text h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 1.75rem;
            font-weight: 700;
            color: #fafafa;
            margin: 0;
        }
        
        .health-header-text p {
            font-family: 'Outfit', sans-serif;
            font-size: 0.9rem;
            color: #71717a;
            margin: 4px 0 0 0;
        }
        
        .health-box {
            background: #1c1c1f;
            border: 1px solid #27272a;
            border-radius: 16px;
            padding: 24px;
            margin: 16px 0;
            transition: all 0.2s ease;
        }
        
        .health-box:hover {
            background: #222225;
            border-color: #3f3f46;
        }
        
        .health-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.1rem;
            font-weight: 600;
            color: #fafafa;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid #27272a;
        }
        
        .health-detail {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #1f1f23;
        }
        
        .health-detail:last-child {
            border-bottom: none;
        }
        
        .health-label {
            font-family: 'Outfit', sans-serif;
            color: #71717a;
            font-size: 0.875rem;
        }
        
        .health-value {
            font-family: 'IBM Plex Mono', monospace;
            font-weight: 600;
            color: #fafafa;
            font-size: 0.9rem;
        }
        
        .health-score-container {
            text-align: center;
            padding: 32px;
        }
        
        .health-score-ring {
            width: 160px;
            height: 160px;
            border-radius: 50%;
            background: conic-gradient(#10b981 var(--score-pct), #27272a 0);
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 16px;
            position: relative;
        }
        
        .health-score-inner {
            width: 130px;
            height: 130px;
            border-radius: 50%;
            background: #1c1c1f;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        
        .health-score-value {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 2.5rem;
            font-weight: 700;
            color: #10b981;
        }
        
        .health-score-label {
            font-family: 'Outfit', sans-serif;
            font-size: 0.8rem;
            color: #71717a;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Back button
    if st.button("← Back to Dashboard"):
        st.session_state.current_page = "dashboard"
        st.rerun()
    
    # Premium Header
    st.markdown('''
        <div class="health-header">
            <div class="health-logo" style="font-size: 18px; font-weight: 700; color: white;">+</div>
            <div class="health-header-text">
                <h1>Vehicle Health Dashboard</h1>
                <p>Real-time diagnostics and predictive maintenance</p>
            </div>
        </div>
    ''', unsafe_allow_html=True)
    
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

# Sidebar with Premium Styling
with st.sidebar:
    # Premium Logo Header
    st.markdown('''
        <div style="display: flex; align-items: center; gap: 12px; padding: 8px 0 20px 0; margin-bottom: 8px; border-bottom: 1px solid #27272a;">
            <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: 700; color: white; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);">VC</div>
            <div>
                <div style="font-family: \'Outfit\', sans-serif; font-size: 1.2rem; font-weight: 700; color: #fafafa; letter-spacing: -0.02em;">VehicleCare</div>
                <div style="font-family: \'Outfit\', sans-serif; font-size: 0.7rem; color: #71717a; text-transform: uppercase; letter-spacing: 0.1em;">AI • Predictive</div>
            </div>
        </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('''
        <div style="font-family: \'Outfit\', sans-serif; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #71717a; margin: 16px 0 8px 0;">Vehicle Control</div>
    ''', unsafe_allow_html=True)
    vehicle_id = st.text_input("Vehicle ID", value="HERO-MNM-01", label_visibility="collapsed", placeholder="Enter Vehicle ID")
    st.session_state.simulator.vehicle_id = vehicle_id
    
    st.markdown('''
        <div style="font-family: \'Outfit\', sans-serif; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #71717a; margin: 24px 0 8px 0;">Fault Simulation</div>
    ''', unsafe_allow_html=True)
    fault_type = st.selectbox(
        "Simulate Component Failure",
        ["None", "Overheat", "Vibration", "Battery Failure", 
         "Throttle Malfunction", "Engine Misfire", 
         "Fuel System Issue", "Cooling System Failure"],
        index=0,
        label_visibility="collapsed"
    )
    
    # Fault status indicator
    fault_active = False
    if fault_type == "Overheat":
        st.session_state.simulator.inject_fault("overheat")
        fault_active = True
    elif fault_type == "Vibration":
        st.session_state.simulator.inject_fault("vibration")
        fault_active = True
    elif fault_type == "Battery Failure":
        st.session_state.simulator.inject_fault("battery_failure")
        fault_active = True
    elif fault_type == "Throttle Malfunction":
        st.session_state.simulator.inject_fault("throttle_malfunction")
        fault_active = True
    elif fault_type == "Engine Misfire":
        st.session_state.simulator.inject_fault("engine_misfire")
        fault_active = True
    elif fault_type == "Fuel System Issue":
        st.session_state.simulator.inject_fault("fuel_system")
        fault_active = True
    elif fault_type == "Cooling System Failure":
        st.session_state.simulator.inject_fault("cooling_system")
        fault_active = True
    else:
        st.session_state.simulator.inject_fault(None)
    
    # Premium fault status indicator
    if fault_active:
        st.markdown(f'''
            <div style="display: flex; align-items: center; gap: 8px; padding: 10px 12px; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 8px; margin: 8px 0;">
                <span style="width: 8px; height: 8px; border-radius: 50%; background: #ef4444; animation: pulse 2s infinite;"></span>
                <span style="font-family: \'Outfit\', sans-serif; font-size: 0.8rem; color: #fca5a5; font-weight: 500;">Fault Active</span>
            </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('''
            <div style="display: flex; align-items: center; gap: 8px; padding: 10px 12px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; margin: 8px 0;">
                <span style="width: 8px; height: 8px; border-radius: 50%; background: #10b981;"></span>
                <span style="font-family: \'Outfit\', sans-serif; font-size: 0.8rem; color: #34d399; font-weight: 500;">Normal Operation</span>
            </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('''
        <div style="font-family: \'Outfit\', sans-serif; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #71717a; margin: 24px 0 8px 0;">Dashboard Controls</div>
    ''', unsafe_allow_html=True)
    auto_update = st.checkbox("Auto Update", value=st.session_state.auto_update)
    st.session_state.auto_update = auto_update
    
    # Update interval dropdown
    interval_options = {
        "1 second": 1,
        "2 seconds": 2,
        "5 seconds": 5,
        "10 seconds": 10,
        "30 seconds": 30,
        "1 minute": 60,
        "2 minutes": 120,
        "5 minutes": 300
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
    
    if st.button("Clear History", use_container_width=True):
        st.session_state.readings_history = []
        st.session_state.anomalies_detected = []
        st.rerun()
    
    # Status Section with premium styling
    st.markdown('''
        <div style="font-family: \'Outfit\', sans-serif; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #71717a; margin: 24px 0 12px 0;">System Status</div>
    ''', unsafe_allow_html=True)
    
    readings_count = len(st.session_state.readings_history)
    anomalies_count = len(st.session_state.anomalies_detected)
    
    st.markdown(f'''
        <div style="display: grid; gap: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 12px; background: #1c1c1f; border: 1px solid #27272a; border-radius: 8px;">
                <span style="font-family: \'Outfit\', sans-serif; font-size: 0.8rem; color: #a1a1aa;">Readings</span>
                <span style="font-family: \'IBM Plex Mono\', monospace; font-size: 0.9rem; color: #fafafa; font-weight: 600;">{readings_count}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 12px; background: #1c1c1f; border: 1px solid {'rgba(239, 68, 68, 0.3)' if anomalies_count > 0 else '#27272a'}; border-radius: 8px;">
                <span style="font-family: \'Outfit\', sans-serif; font-size: 0.8rem; color: #a1a1aa;">Anomalies</span>
                <span style="font-family: \'IBM Plex Mono\', monospace; font-size: 0.9rem; color: {'#f87171' if anomalies_count > 0 else '#fafafa'}; font-weight: 600;">{anomalies_count}</span>
            </div>
        </div>
    ''', unsafe_allow_html=True)
    
    # Quick Access
    st.markdown('''
        <div style="font-family: \'Outfit\', sans-serif; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #71717a; margin: 24px 0 12px 0;">Quick Access</div>
    ''', unsafe_allow_html=True)
    
    if st.button("Health Dashboard", use_container_width=True):
        st.session_state.current_page = "health_dashboard"
        st.rerun()
    
    if st.session_state.appointments:
        if st.button(f"Appointments ({len(st.session_state.appointments)})", use_container_width=True):
            st.session_state.current_page = "appointments"
            st.rerun()

# Main dashboard - Page routing
# Check if there's a notification to show - AUTO-BOOKING ON ANOMALY DETECTION
if st.session_state.show_notification and st.session_state.current_page == "dashboard":
    # Auto-navigate directly to auto-booking progress when anomaly detected
    st.session_state.auto_booking_triggered = False  # Reset for new booking
    st.session_state.auto_booking_complete = False
    st.session_state.calling_centers_progress = []
    st.session_state.current_page = "auto_booking_progress"
    st.rerun()

# Route to appropriate page
if st.session_state.current_page == "issue_details":
    render_issue_details_page()
    st.stop()
elif st.session_state.current_page == "auto_booking_progress":
    render_auto_booking_progress_page()
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

# Default: Full dashboard with premium header
st.markdown('''
    <div style="display: flex; align-items: center; justify-content: space-between; padding: 0 0 24px 0; margin-bottom: 24px; border-bottom: 1px solid #27272a;">
        <div style="display: flex; align-items: center; gap: 16px;">
            <div style="width: 56px; height: 56px; background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%); border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 18px; font-weight: 700; color: white; box-shadow: 0 8px 24px rgba(16, 185, 129, 0.3);">VC</div>
            <div>
                <h1 style="font-family: \'Outfit\', sans-serif; font-size: 1.75rem; font-weight: 700; color: #fafafa; margin: 0; letter-spacing: -0.02em;">Predictive Maintenance</h1>
                <p style="font-family: \'Outfit\', sans-serif; font-size: 0.9rem; color: #71717a; margin: 4px 0 0 0;">Real-time vehicle telemetry monitoring and AI anomaly detection</p>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); padding: 8px 16px; border-radius: 24px;">
            <span style="width: 8px; height: 8px; border-radius: 50%; background: #10b981; animation: pulse 2s ease-in-out infinite;"></span>
            <span style="font-family: \'Outfit\', sans-serif; font-size: 0.8rem; color: #34d399; font-weight: 500;">Live Monitoring</span>
        </div>
    </div>
''', unsafe_allow_html=True)

# Auto-update logic - Generate new data based on interval
if st.session_state.auto_update:
    current_time = time.time()
    
    # Initialize last_update_time if not set
    if "last_update_time" not in st.session_state:
        st.session_state.last_update_time = current_time
    
    time_since_last_update = current_time - st.session_state.last_update_time
    
    # Check if it's time to generate new data
    if time_since_last_update >= st.session_state.update_interval:
        # Sync detector history with session state history before detection
        if hasattr(st.session_state.detector, 'sync_history'):
            st.session_state.detector.sync_history(st.session_state.readings_history)
        
        # Generate new reading
        reading = st.session_state.simulator.generate_reading()
        anomaly = st.session_state.detector.detect_anomaly(reading)
        score = st.session_state.detector.get_anomaly_score(reading)
        
        reading["anomaly"] = anomaly
        reading["anomaly_score"] = score
        
        # Update latest reading for display
        st.session_state.latest_reading = reading
        
        # Add to history
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
        
        # Sync detector history after adding
        if hasattr(st.session_state.detector, 'sync_history'):
            st.session_state.detector.sync_history(st.session_state.readings_history)
    
    # Calculate time until next update
    time_until_next = max(0, st.session_state.update_interval - time_since_last_update)
    
    # Show refresh status
    st.info(f"Auto-updating every {st.session_state.update_interval}s | Next update in {int(time_until_next)}s | Total readings: {len(st.session_state.readings_history)}")

# Display latest anomaly alert with notification banner (compact)
if st.session_state.anomalies_detected:
    latest_anomaly = st.session_state.anomalies_detected[-1]
    
    # Show compact notification banner only
    st.markdown("---")
    col1, col2 = st.columns([4, 1])
    with col1:
        st.error("🚨 **ANOMALY DETECTED** - Vehicle issue identified by predictive analysis")
    with col2:
        if st.button("View Details", type="primary"):
            st.session_state.current_issue = latest_anomaly
            st.session_state.current_page = "issue_details"
            st.rerun()
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
        
        # Update layout with dark theme styling
        fig.update_layout(
            height=1000,
            showlegend=False,
            title_text="Vehicle Telemetry Dashboard",
            title_x=0.5,
            margin=dict(l=50, r=50, t=80, b=50),
            paper_bgcolor='rgba(28, 28, 31, 1)',
            plot_bgcolor='rgba(24, 24, 27, 1)',
            font=dict(color='#a1a1aa', family='Outfit, sans-serif'),
            title_font=dict(color='#fafafa', size=18, family='Outfit, sans-serif')
        )
        
        # Update axes for dark theme
        fig.update_xaxes(
            gridcolor='#27272a',
            linecolor='#3f3f46',
            tickfont=dict(color='#71717a'),
            title_font=dict(color='#a1a1aa')
        )
        fig.update_yaxes(
            gridcolor='#27272a',
            linecolor='#3f3f46',
            tickfont=dict(color='#71717a'),
            title_font=dict(color='#a1a1aa')
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

# ============================================
# AUTO-REFRESH MECHANISM (at end of page)
# ============================================
# This runs AFTER all content is rendered, so dashboard is visible
if st.session_state.auto_update:
    import time as time_module
    
    # Wait for the update interval, then refresh
    # Using a shorter sleep to be responsive
    sleep_time = min(st.session_state.update_interval, 1.0)
    time_module.sleep(sleep_time)
    st.rerun()

