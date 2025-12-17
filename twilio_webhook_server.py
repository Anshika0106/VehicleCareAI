"""
Twilio Webhook Server
FastAPI server to handle Twilio voice webhooks for real-time call handling.
Run this separately to handle incoming Twilio callbacks during live calls.
"""

import os
import json
import asyncio
import base64
from typing import Dict, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

# Import our voice booking components
from voice_booking_agent import (
    AutomatedBookingSystem,
    VoiceBookingConversationAgent,
    AzureSpeechManager,
    BookingRequest,
    BookingStatus
)


# Configuration from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "eastus")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")

# Active call sessions
active_sessions: Dict[str, dict] = {}

# Speech manager instance
speech_manager: Optional[AzureSpeechManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI."""
    global speech_manager
    
    # Initialize speech manager on startup
    if AZURE_SPEECH_KEY and AZURE_SPEECH_REGION:
        speech_manager = AzureSpeechManager(AZURE_SPEECH_KEY, AZURE_SPEECH_REGION)
        print("✓ Azure Speech Manager initialized")
    else:
        print("⚠ Azure Speech credentials not configured")
    
    yield
    
    # Cleanup on shutdown
    active_sessions.clear()
    print("✓ Webhook server shut down")


app = FastAPI(
    title="VehicleCare Voice Booking Webhook Server",
    description="Handles Twilio webhooks for automated voice booking calls",
    version="1.0.0",
    lifespan=lifespan
)


class BookingSessionData(BaseModel):
    """Data model for storing booking session information."""
    booking_id: str
    customer_name: str
    customer_phone: str
    customer_email: str
    vehicle_id: str
    issue_type: str
    issue_description: str
    severity: str
    preferred_date: str
    preferred_time: str
    service_center_name: str


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "VehicleCare Voice Booking Webhook Server",
        "active_sessions": len(active_sessions)
    }


@app.post("/voice/outbound/{booking_id}")
async def handle_outbound_call(booking_id: str, request: Request):
    """
    Handle the initial outbound call connection.
    Twilio calls this URL when the call is answered.
    """
    form_data = await request.form()
    call_status = form_data.get("CallStatus", "")
    
    print(f"📞 Outbound call {booking_id}: {call_status}")
    
    if booking_id not in active_sessions:
        # Return a basic response if session not found
        twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">
        We're sorry, but this call session has expired. Please try booking again.
    </Say>
    <Hangup/>
</Response>"""
        return PlainTextResponse(content=twiml, media_type="text/xml")
    
    session = active_sessions[booking_id]
    agent = session.get('agent')
    
    if agent:
        # Get the opening message
        opening_message = agent.get_opening_message()
        
        # Generate TwiML response
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather input="speech" action="/voice/gather/{booking_id}" method="POST" 
            speechTimeout="auto" language="en-US">
        <Say voice="Polly.Joanna">{opening_message}</Say>
    </Gather>
    <Say voice="Polly.Joanna">I didn't catch that. Let me try again.</Say>
    <Redirect>/voice/outbound/{booking_id}</Redirect>
</Response>"""
    else:
        twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">
        Hello, I'm calling from VehicleCare AI to schedule a service appointment.
    </Say>
    <Gather input="speech" action="/voice/gather/{booking_id}" method="POST" 
            speechTimeout="auto" language="en-US">
        <Say voice="Polly.Joanna">How can you assist me today?</Say>
    </Gather>
</Response>"""
    
    return PlainTextResponse(content=twiml, media_type="text/xml")


@app.post("/voice/gather/{booking_id}")
async def handle_speech_gather(booking_id: str, request: Request):
    """
    Handle gathered speech input from the service center.
    Processes the speech and generates the next response.
    """
    form_data = await request.form()
    speech_result = form_data.get("SpeechResult", "")
    confidence = form_data.get("Confidence", "0")
    
    print(f"🎤 Received speech [{booking_id}]: {speech_result} (confidence: {confidence})")
    
    if booking_id not in active_sessions:
        twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">Thank you for your time. Goodbye.</Say>
    <Hangup/>
</Response>"""
        return PlainTextResponse(content=twiml, media_type="text/xml")
    
    session = active_sessions[booking_id]
    agent = session.get('agent')
    
    if agent and speech_result:
        # Store the transcript
        if 'transcript' not in session:
            session['transcript'] = []
        session['transcript'].append({
            'speaker': 'service_center',
            'text': speech_result,
            'timestamp': datetime.now().isoformat()
        })
        
        # Get AI response
        try:
            response_text = await agent.process_response(speech_result)
            
            session['transcript'].append({
                'speaker': 'ai',
                'text': response_text,
                'timestamp': datetime.now().isoformat()
            })
            
            # Check if booking is confirmed
            if agent.is_booking_confirmed():
                session['status'] = BookingStatus.CONFIRMED
                session['confirmation'] = agent.get_confirmation_details()
                
                # End the call gracefully
                twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">{response_text}</Say>
    <Say voice="Polly.Joanna">Thank you so much for your help. Have a great day!</Say>
    <Hangup/>
</Response>"""
            else:
                # Continue the conversation
                twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather input="speech" action="/voice/gather/{booking_id}" method="POST" 
            speechTimeout="auto" language="en-US">
        <Say voice="Polly.Joanna">{response_text}</Say>
    </Gather>
    <Say voice="Polly.Joanna">I'm sorry, I didn't catch that. Could you please repeat?</Say>
    <Redirect>/voice/gather/{booking_id}</Redirect>
</Response>"""
                
        except Exception as e:
            print(f"❌ Error processing speech: {e}")
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">I apologize, I'm having some technical difficulties. 
    Could you please repeat that?</Say>
    <Gather input="speech" action="/voice/gather/{booking_id}" method="POST" 
            speechTimeout="auto" language="en-US"/>
</Response>"""
    else:
        # No speech detected
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">I'm sorry, I didn't catch that.</Say>
    <Gather input="speech" action="/voice/gather/{booking_id}" method="POST" 
            speechTimeout="auto" language="en-US">
        <Say voice="Polly.Joanna">Could you please repeat that?</Say>
    </Gather>
</Response>"""
    
    return PlainTextResponse(content=twiml, media_type="text/xml")


@app.post("/voice/status/{booking_id}")
async def handle_call_status(booking_id: str, request: Request):
    """
    Handle call status updates from Twilio.
    """
    form_data = await request.form()
    call_status = form_data.get("CallStatus", "")
    call_duration = form_data.get("CallDuration", "0")
    
    print(f"📊 Call status [{booking_id}]: {call_status} (duration: {call_duration}s)")
    
    if booking_id in active_sessions:
        session = active_sessions[booking_id]
        session['call_status'] = call_status
        session['call_duration'] = call_duration
        
        if call_status in ['completed', 'failed', 'busy', 'no-answer']:
            session['ended_at'] = datetime.now().isoformat()
            
            if call_status == 'completed' and session.get('status') != BookingStatus.CONFIRMED:
                # Call ended without confirmation
                session['status'] = BookingStatus.FAILED
    
    return {"status": "received"}


@app.websocket("/voice/stream/{booking_id}")
async def handle_media_stream(websocket: WebSocket, booking_id: str):
    """
    Handle bidirectional media streaming for real-time voice processing.
    This enables more natural conversation with lower latency.
    """
    await websocket.accept()
    
    print(f"🔊 Media stream connected [{booking_id}]")
    
    if booking_id not in active_sessions:
        await websocket.close()
        return
    
    session = active_sessions[booking_id]
    audio_buffer = b""
    
    try:
        while True:
            # Receive message from Twilio
            message = await websocket.receive_text()
            data = json.loads(message)
            
            event_type = data.get("event")
            
            if event_type == "connected":
                print(f"🔗 Stream connected [{booking_id}]")
                
            elif event_type == "start":
                stream_sid = data.get("streamSid")
                session['stream_sid'] = stream_sid
                print(f"▶️ Stream started [{booking_id}]: {stream_sid}")
                
            elif event_type == "media":
                # Incoming audio from the call
                payload = data.get("media", {}).get("payload", "")
                audio_data = base64.b64decode(payload)
                audio_buffer += audio_data
                
                # Process audio in chunks (e.g., every 1 second of audio)
                # Twilio sends audio at 8kHz mono, 20ms chunks
                if len(audio_buffer) >= 8000:  # ~1 second of audio
                    # In production: send to speech-to-text
                    # recognized_text = await process_audio_to_text(audio_buffer)
                    audio_buffer = b""
                    
            elif event_type == "stop":
                print(f"⏹️ Stream stopped [{booking_id}]")
                break
                
    except WebSocketDisconnect:
        print(f"🔌 Stream disconnected [{booking_id}]")
    except Exception as e:
        print(f"❌ Stream error [{booking_id}]: {e}")
    finally:
        await websocket.close()


@app.post("/session/create")
async def create_booking_session(data: BookingSessionData):
    """
    Create a new booking session.
    Called by the main app before initiating a call.
    """
    from datetime import datetime as dt
    
    # Parse the date
    try:
        preferred_date = dt.strptime(data.preferred_date, "%Y-%m-%d")
    except:
        preferred_date = dt.now()
    
    # Create booking request
    booking_request = BookingRequest(
        customer_name=data.customer_name,
        customer_phone=data.customer_phone,
        customer_email=data.customer_email,
        vehicle_id=data.vehicle_id,
        issue_type=data.issue_type,
        issue_description=data.issue_description,
        severity=data.severity,
        preferred_date=preferred_date,
        preferred_time=data.preferred_time,
        service_center_phone="",  # Will be set by directory lookup
        service_center_name=data.service_center_name
    )
    
    # Create conversation agent
    agent = VoiceBookingConversationAgent(
        GOOGLE_API_KEY,
        booking_request
    )
    
    # Store session
    active_sessions[data.booking_id] = {
        'agent': agent,
        'request': booking_request,
        'status': BookingStatus.PENDING,
        'created_at': datetime.now().isoformat(),
        'transcript': []
    }
    
    return {
        "status": "created",
        "booking_id": data.booking_id,
        "message": "Session ready for call"
    }


@app.get("/session/{booking_id}")
async def get_session_status(booking_id: str):
    """
    Get the status of a booking session.
    """
    if booking_id not in active_sessions:
        return {"status": "not_found"}
    
    session = active_sessions[booking_id]
    
    return {
        "status": session.get('status', BookingStatus.PENDING).value,
        "call_status": session.get('call_status'),
        "confirmation": session.get('confirmation'),
        "transcript": session.get('transcript', []),
        "created_at": session.get('created_at'),
        "ended_at": session.get('ended_at')
    }


@app.delete("/session/{booking_id}")
async def delete_session(booking_id: str):
    """
    Delete a booking session.
    """
    if booking_id in active_sessions:
        del active_sessions[booking_id]
        return {"status": "deleted"}
    return {"status": "not_found"}


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("VehicleCare Voice Booking Webhook Server")
    print("=" * 60)
    print("\nThis server handles Twilio webhooks for voice calls.")
    print("For production, expose this server via ngrok or similar.")
    print("\nExample: ngrok http 8000")
    print("\nThen configure your Twilio webhook URLs to use the ngrok URL.")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

