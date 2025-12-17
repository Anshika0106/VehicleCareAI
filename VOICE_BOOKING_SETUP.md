# 🤖 VehicleCare AI - Automated Voice Booking System

This document explains how to set up and use the automated appointment booking feature that calls service centers on behalf of users.

## Overview

The Voice Booking System uses:
- **LangChain + Google Gemini** - Conversational AI agent that handles the booking dialog
- **Azure Speech SDK** - Text-to-Speech and Speech-to-Text for natural voice
- **Twilio** - Telephony platform for making actual phone calls

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        VehicleCare AI                            │
├─────────────────────────────────────────────────────────────────┤
│  Streamlit App (app.py)                                         │
│    └── User triggers "Auto-Book via AI Call"                    │
│                                                                  │
│  Voice Booking Agent (voice_booking_agent.py)                   │
│    ├── BookingRequest → Contains customer & issue details       │
│    ├── VoiceBookingConversationAgent → LangChain AI Dialog      │
│    ├── AzureSpeechManager → TTS/STT conversion                  │
│    ├── TwilioCallManager → Phone call management                │
│    └── AutomatedBookingSystem → Orchestrates everything         │
│                                                                  │
│  Twilio Webhook Server (twilio_webhook_server.py)               │
│    └── Handles real-time call events & audio streaming          │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start (Simulation Mode)

You can test the AI booking system **without making real phone calls** in simulation mode.

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Google Gemini API Key

```bash
export GOOGLE_API_KEY="your-gemini-api-key"
```

Or create a `.env` file:
```
GOOGLE_API_KEY=your-gemini-api-key
```

### 3. Run the App

```bash
streamlit run app.py
```

### 4. Test the Feature
1. Simulate a vehicle fault (use sidebar controls)
2. When anomaly is detected, click "View Details"
3. Click "Schedule Service Appointment"
4. Click "📞 Auto-Book via AI Call"
5. Watch the AI simulate the booking conversation!

---

## Production Setup (Live Phone Calls)

For actual phone calls to service centers, you need to configure all three services.

### 1. Google Gemini Configuration

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create an API key
3. Set environment variable:
   ```bash
   export GOOGLE_API_KEY="your-gemini-api-key"
   ```

### 2. Azure Speech SDK Configuration

1. Create an [Azure account](https://azure.microsoft.com/)
2. Create a Speech resource:
   - Go to [Azure Portal](https://portal.azure.com/)
   - Click "Create a resource"
   - Search for "Speech"
   - Create a Speech resource
3. Get your key and region from the resource's "Keys and Endpoint" page
4. Set environment variables:
   ```bash
   export AZURE_SPEECH_KEY="your-speech-key"
   export AZURE_SPEECH_REGION="eastus"  # or your region
   ```

### 3. Twilio Configuration

1. Sign up at [Twilio](https://www.twilio.com/)
2. Get your Account SID and Auth Token from [Twilio Console](https://console.twilio.com/)
3. Purchase a phone number with voice capabilities
4. Set environment variables:
   ```bash
   export TWILIO_ACCOUNT_SID="your-account-sid"
   export TWILIO_AUTH_TOKEN="your-auth-token"
   export TWILIO_PHONE_NUMBER="+15551234567"
   ```

### 4. Webhook Server Setup

For Twilio to handle real-time call events, you need a publicly accessible webhook server.

#### For Local Development (using ngrok):

1. Install ngrok: https://ngrok.com/download
2. Start the webhook server:
   ```bash
   python twilio_webhook_server.py
   ```
3. In another terminal, expose it:
   ```bash
   ngrok http 8000
   ```
4. Copy the HTTPS URL (e.g., `https://abc123.ngrok.io`)
5. Set the environment variable:
   ```bash
   export WEBHOOK_BASE_URL="https://abc123.ngrok.io"
   ```

#### For Production:
Deploy `twilio_webhook_server.py` to a cloud platform:
- **Heroku**: Add to Procfile: `web: uvicorn twilio_webhook_server:app --host 0.0.0.0 --port $PORT`
- **AWS Lambda**: Use Mangum adapter
- **Google Cloud Run**: Containerize with Docker
- **Railway/Render**: Direct Python deployment

---

## How It Works

### 1. Booking Request Created
When user clicks "Auto-Book via AI Call", a `BookingRequest` is created with:
- Customer info (name, phone, email)
- Vehicle details and issue
- Preferred appointment date/time
- Service center information

### 2. AI Agent Initialized
The `VoiceBookingConversationAgent` is initialized with:
- System prompt containing all booking details
- Instructions for professional phone conversation
- Guidelines for confirming booking details

### 3. Call Initiated (or Simulated)
- **Simulation Mode**: AI simulates both sides of the conversation
- **Live Mode**: Twilio initiates a real phone call

### 4. Conversation Flow
```
AI: "Hello, I'm calling from VehicleCare AI to schedule 
     a service appointment for [customer name]..."

Service Center: "Hello, thank you for calling..."

AI: [Processes response with LangChain]
    "Great! We need an appointment for a battery issue. 
     Do you have availability on [date] at [time]?"

Service Center: "Let me check... yes, we have that slot."

AI: [Detects confirmation, extracts details]
    "Perfect! Can you confirm the appointment details?"
    
... continues until booking confirmed or failed ...
```

### 5. Result Returned
After the call:
- Confirmation number extracted
- Scheduled date/time recorded
- Call transcript saved
- Appointment added to user's list

---

## Configuration Reference

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes | Google API key for Gemini |
| `AZURE_SPEECH_KEY` | Yes* | Azure Speech SDK key |
| `AZURE_SPEECH_REGION` | Yes* | Azure region (e.g., "eastus") |
| `TWILIO_ACCOUNT_SID` | No** | Twilio Account SID |
| `TWILIO_AUTH_TOKEN` | No** | Twilio Auth Token |
| `TWILIO_PHONE_NUMBER` | No** | Twilio phone number |
| `WEBHOOK_BASE_URL` | No** | Webhook server URL |

\* Required for voice synthesis (simulation works without it)
\** Required only for live phone calls

### Files Added

| File | Purpose |
|------|---------|
| `voice_booking_agent.py` | Main voice booking system |
| `twilio_webhook_server.py` | FastAPI server for Twilio webhooks |
| `config.py` | Configuration management |

---

## Customization

### Change AI Voice
In `voice_booking_agent.py`, modify the voice name:
```python
self.speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
```

Available voices: [Azure Neural Voices](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts)

### Modify Conversation Style
Edit the system prompt in `VoiceBookingConversationAgent._create_system_prompt()` to change:
- Tone and formality
- Conversation flow
- Information to collect/provide

### Add Service Centers
Update `SERVICE_CENTER_DIRECTORY` in `voice_booking_agent.py`:
```python
SERVICE_CENTER_DIRECTORY = {
    "Your Service Center": {
        "phone": "+1-555-0100",
        "address": "123 Main St",
        "hours": "8:00 AM - 6:00 PM"
    }
}
```

---

## Troubleshooting

### "API keys not configured" Warning
- Running in simulation mode is fine for testing
- Set environment variables for full functionality

### Call Not Connecting
- Verify Twilio credentials
- Check phone number format (E.164: +15551234567)
- Ensure webhook server is running and accessible

### Speech Not Working
- Verify Azure Speech credentials
- Check region matches your resource
- Ensure audio format is supported

### AI Responses Slow
- Gemini 1.5 Pro is optimized for quality
- Consider using Gemini 1.5 Flash for faster responses
- Adjust temperature parameter for creativity vs consistency

---

## Security Considerations

1. **Never commit API keys** to version control
2. Use environment variables or secrets management
3. Enable call recording only with consent
4. Implement rate limiting for production
5. Monitor usage and costs for all APIs

---

## Cost Estimation

| Service | Cost |
|---------|------|
| Google Gemini 1.5 Pro | ~$0.00125-0.005 per booking conversation (very affordable!) |
| Azure Speech | ~$0.01 per 1000 characters |
| Twilio Voice | ~$0.014/min outbound calls |

Estimated cost per automated booking: **~$0.05-0.10**

---

## Support

For issues or questions:
1. Check this documentation
2. Run `python config.py` to verify configuration
3. Review call transcripts for conversation issues
4. Check webhook server logs for call handling issues

