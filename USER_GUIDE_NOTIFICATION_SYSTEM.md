# VehicleCare AI - Anomaly Notification & Booking System User Guide

## Overview
The VehicleCare AI application now includes a comprehensive notification and booking system that automatically alerts you when vehicle anomalies are detected and guides you through scheduling service appointments.

## How to Test the New Features

### Step 1: Start the Application
The Streamlit app should already be running. If not, run:
```bash
cd VehicleCareAI
streamlit run app.py
```

### Step 2: Trigger an Anomaly Detection

#### Option A: Using Auto-Update (Recommended)
1. **Enable Auto Update** in the sidebar (checkbox should be checked)
2. **Select a Fault Type** from the "Simulate Component Failure" dropdown
   - Try "Battery Failure" for a good example
3. **Wait a few seconds** - The system will automatically:
   - Detect the anomaly
   - Navigate to the "Issue Detected" page
   - Display the notification

#### Option B: Manual Generation
1. **Select a Fault Type** from the sidebar dropdown
2. **Click "Generate New Reading"** button
3. If an anomaly is detected, you'll see a notification banner
4. **Click "View Details"** on the notification banner

### Step 3: Review the Issue Details Page

You should now see a page similar to **Wireframe 2** with:

- ✅ **Issue Detected** section
  - Issue title (e.g., "Battery Health Deterioration")
  - Detailed description from the predictive model
  
- ✅ **Issue Details** section
  - Severity Level: Critical / Major / Minor
  
- ✅ **Recommended Action** section
  - Specific action to take

- ✅ **Schedule Service** button

### Step 4: Schedule a Service Appointment

1. **Click "Schedule Service"** button
2. Fill out the appointment form:
   - **Service Center**: Select from 4 available locations
   - **Service Type**: Auto-suggested based on detected issue
   - **Date**: Pick a date (1-30 days from today)
   - **Time**: Select from available time slots (8 AM - 5 PM)
   - **Customer Info**: Name, Phone, Email (pre-filled with demo data)
3. **Click "Confirm Booking"**

### Step 5: View Appointment Confirmation

You should now see a confirmation page (**Wireframe 1**) showing:

- ✅ **Status**: Confirmed
- ✅ **Appointment Details**:
  - Service Center name
  - Service Type
  - Date (formatted as "Wednesday, December 25, 2025")
  - Time
- ✅ **Arrival Instructions**
- ✅ **Action Buttons**:
  - "View Appointment"
  - "Back to Dashboard"

### Step 6: Access Vehicle Health Dashboard

From the main dashboard:

1. **Click "Vehicle Health Dashboard"** button in the sidebar
2. View the simplified health dashboard (**Wireframe 3**) showing:
   - Vehicle ID (VIN)
   - Health Score (0-100%)
   - Predicted Issue
   - Risk Level
3. **Click "View Details"** to return to issue details or full dashboard

### Step 7: Manage Appointments

1. **Click "View Appointments"** button in sidebar (appears when you have appointments)
2. View all scheduled appointments with full details
3. Expand any appointment to see:
   - Status, Service Center, Service Type
   - Date, Time, Issue
   - Customer information

## Testing Different Fault Types

Try different fault types to see various issue notifications:

### Battery Failure
- **Issue**: "Battery Health Deterioration" or "Battery Failure Critical"
- **Service Type**: Battery Diagnosis & Replacement
- **Severity**: Major or Critical
- **Health Impact**: -20 to -30 points

### Overheat
- **Issue**: "Coolant System Failure"
- **Service Type**: Cooling System Inspection & Repair
- **Severity**: Critical
- **Health Impact**: -25 to -30 points

### Vibration
- **Issue**: "Mechanical Looseness Detected"
- **Service Type**: Vibration Diagnosis & Repair
- **Severity**: Critical
- **Health Impact**: -20 to -25 points

### Engine Misfire
- **Issue**: "Engine Misfire Detected"
- **Service Type**: Engine Inspection & Repair
- **Severity**: Major
- **Health Impact**: -15 to -20 points

### Throttle Malfunction
- **Issue**: "Throttle System Malfunction"
- **Service Type**: Throttle System Repair
- **Severity**: Critical
- **Health Impact**: -15 to -25 points

### Fuel System Issue
- **Issue**: "Fuel System Malfunction"
- **Service Type**: Fuel System Inspection & Repair
- **Severity**: Major
- **Health Impact**: -15 to -20 points

### Cooling System Failure
- **Issue**: "Cooling System Failure"
- **Service Type**: Cooling System Inspection & Repair
- **Severity**: Major or Critical
- **Health Impact**: -15 to -25 points

## Navigation Flow

```
Main Dashboard
    ↓ (Anomaly Detected - Auto)
Issue Details Page
    ↓ (Click "Schedule Service")
Schedule Service Page
    ↓ (Click "Confirm Booking")
Confirmation Page
    ↓ (Click "Back to Dashboard" or "View Appointment")
Main Dashboard / Appointments Page
```

## Quick Access Features

### Sidebar Quick Access
- **Vehicle Health Dashboard** - View simplified health metrics
- **View Appointments (X)** - See all scheduled appointments (X = count)

### Notification Banner (Main Dashboard)
- Appears when anomalies are detected
- **View Details** button - Quick access to issue details
- Shows latest anomaly alert with full recommendation

### Back Navigation
- **← Back** button on all sub-pages
- **Back to Dashboard** button on confirmation page
- Easy navigation throughout the system

## Tips for Best Experience

1. **Enable Auto Update** for continuous monitoring
2. **Adjust Update Interval** to control detection frequency (5s - 10min)
3. **Clear History** periodically to reset the system
4. **Try Different Faults** to see various notification types
5. **Check Appointments** to review all scheduled services

## Features Preserved

All original dashboard features remain functional:

✅ Real-time telemetry monitoring
✅ Anomaly detection with LSTM + XGBoost
✅ Interactive charts (6 sensor graphs)
✅ Anomaly history table
✅ Vehicle simulation controls
✅ Fault injection for testing
✅ Auto-update functionality

## Health Score Calculation

The health score (0-100%) is calculated based on:

- **Temperature** (Max -30 points)
  - Normal: ≤105°C
  - Warning: >105°C
  - Critical: >120°C

- **Vibration** (Max -25 points)
  - Normal: ≤0.4g
  - Warning: >0.4g
  - Critical: >1.0g

- **Battery Voltage** (Max -20 points)
  - Normal: 13.5-14.5V
  - Low: <13.5V
  - Critical: <11.5V
  - High: >14.5V

- **Engine RPM** (Max -15 points)
  - Normal: 800-3000 RPM
  - Low: <800 RPM
  - High: >3000 RPM

## Troubleshooting

### No Anomaly Detected
- Ensure a fault type is selected (not "None")
- Try "Battery Failure" or "Overheat" for guaranteed detection
- Wait for auto-update to generate new readings

### Page Not Changing
- Check browser console for errors
- Refresh the page (Ctrl+R or Cmd+R)
- Restart Streamlit server if needed

### Back Button Not Working
- Click the button again
- Refresh the page
- Check session state in sidebar status

## Support

For issues or questions:
1. Check the logs in the terminal running Streamlit
2. Review the NOTIFICATION_SYSTEM_SUMMARY.md for technical details
3. Verify all dependencies are installed (requirements.txt)

---

**Enjoy the new notification and booking system!** 🚗✨

