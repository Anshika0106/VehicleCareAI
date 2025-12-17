# Anomaly Notification & Booking System - Implementation Summary

## Overview
Implemented a comprehensive notification and booking system based on the provided wireframes that alerts users when anomalies are detected and allows them to schedule service appointments.

## Key Features Implemented

### 1. **Vehicle Health Dashboard** (Wireframe 3)
- Simplified view showing:
  - Vehicle ID (VIN)
  - Health Score (0-100%)
  - Predicted Issue
  - Risk Level (High/Medium/Low)
- Accessible via sidebar "Vehicle Health Dashboard" button
- "View Details" button to see more information

### 2. **Issue Detection Notification** (Wireframe 2)
- Automatic notification when anomaly is detected
- Structured display with:
  - Issue title (e.g., "Battery Health Deterioration")
  - Detailed description from predictive model
  - Severity level (Critical/Major/Minor)
  - Recommended action
- "Schedule Service" button to book appointment

### 3. **Service Appointment Scheduling**
- Form to schedule service with:
  - Service center selection (4 locations available)
  - Auto-suggested service type based on detected issue
  - Date picker (1-30 days in advance)
  - Time slot selection (8 AM - 5 PM)
  - Customer information fields
- "Confirm Booking" button to finalize appointment

### 4. **Appointment Confirmation** (Wireframe 1)
- Confirmation page showing:
  - Status: Confirmed
  - Service center name
  - Service type
  - Date and time
  - Arrival instructions
- "View Appointment" and "Back to Dashboard" buttons

### 5. **Appointments Management**
- View all scheduled appointments
- Appointment history with full details
- Accessible via sidebar when appointments exist

### 6. **Enhanced Main Dashboard**
- Prominent notification banner when anomalies detected
- "View Details" button on notification banner
- Quick access to issue details from anomaly alerts
- All existing functionality preserved

## New Functions Added

### In `maintenance_agent.py`:
1. `get_issue_details()` - Returns structured issue information (title, description, action)
2. `get_severity_level()` - Determines severity (Critical/Major/Minor)
3. `calculate_health_score()` - Calculates 0-100 health score
4. `get_predicted_issue()` - Returns short issue description
5. `get_risk_level()` - Determines risk level (High/Medium/Low)

### In `app.py`:
1. `render_issue_details_page()` - Issue detection page (Wireframe 2)
2. `render_schedule_service_page()` - Service booking page
3. `render_confirmation_page()` - Appointment confirmation (Wireframe 1)
4. `render_vehicle_health_dashboard()` - Health dashboard view (Wireframe 3)
5. `render_appointments_page()` - Appointments list view

## User Flow

1. **Anomaly Detected** → Auto-navigate to Issue Details page
2. **Issue Details** → User reviews issue information
3. **Schedule Service** → User fills out appointment form
4. **Confirmation** → User sees confirmed appointment details
5. **Dashboard** → Return to monitoring

## Session State Variables Added

- `current_page` - Tracks which page to display
- `current_issue` - Stores the active anomaly details
- `appointments` - List of all scheduled appointments
- `latest_appointment` - Most recent appointment for confirmation page
- `show_notification` - Flag to trigger notification display

## Testing Instructions

1. **Enable Auto Update** in sidebar
2. **Select a fault type** (e.g., "Battery Failure")
3. **Wait for anomaly detection** - System will auto-navigate to Issue Details
4. **Review the issue** - Check severity and recommended action
5. **Click "Schedule Service"** - Fill out the appointment form
6. **Confirm booking** - See the confirmation page
7. **View appointments** - Access via sidebar button

## Service Types Mapped to Issues

- Battery issues → "Battery Diagnosis & Replacement"
- Cooling/Overheating → "Cooling System Inspection & Repair"
- Vibration → "Vibration Diagnosis & Repair"
- Engine misfire → "Engine Inspection & Repair"
- Throttle issues → "Throttle System Repair"
- Fuel system → "Fuel System Inspection & Repair"
- Generic → "General Inspection & Diagnosis"

## Health Score Calculation

The health score (0-100) is calculated by:
- Starting at 100 (perfect health)
- Deducting points for temperature anomalies (up to 30 points)
- Deducting points for vibration issues (up to 25 points)
- Deducting points for battery voltage issues (up to 20 points)
- Deducting points for RPM anomalies (up to 15 points)

## UI Enhancements

- Styled boxes with borders for professional appearance
- Color-coded severity levels
- Prominent notification banners
- Responsive button layouts
- Clean, minimalist design matching wireframes
- Proper spacing and typography

## Integration Points

- Seamlessly integrates with existing anomaly detection system
- Preserves all original dashboard functionality
- Uses existing VehicleSimulator and AnomalyDetector classes
- Extends existing maintenance_agent module

## Future Enhancements (Optional)

- Email/SMS notifications for appointments
- Calendar integration (iCal/Google Calendar)
- Appointment rescheduling/cancellation
- Service center availability checking
- Cost estimates for services
- Service history tracking

