"""
Voice Booking Agent
Automated appointment booking system using LangChain + Azure Speech SDK + Twilio.
This agent calls the service center and books appointments on behalf of the user.
"""

import os
import json
import asyncio
from typing import Dict, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Optional imports - these may not be installed yet
# LangChain with Gemini
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.schema import HumanMessage, AIMessage, SystemMessage
    from langchain.memory import ConversationBufferMemory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatGoogleGenerativeAI = None
    HumanMessage = None
    AIMessage = None
    SystemMessage = None
    ConversationBufferMemory = None

# Azure Speech SDK
try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_SPEECH_AVAILABLE = True
except ImportError:
    AZURE_SPEECH_AVAILABLE = False
    speechsdk = None


class BookingStatus(Enum):
    """Status of the automated booking call."""
    PENDING = "pending"
    CALLING = "calling"
    IN_PROGRESS = "in_progress"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BookingRequest:
    """Data class for booking request details."""
    customer_name: str
    customer_phone: str
    customer_email: str
    vehicle_id: str
    issue_type: str
    issue_description: str
    severity: str
    preferred_date: datetime
    preferred_time: str
    service_center_phone: str
    service_center_name: str


@dataclass
class BookingResult:
    """Data class for booking result."""
    status: BookingStatus
    confirmation_number: Optional[str] = None
    scheduled_date: Optional[str] = None
    scheduled_time: Optional[str] = None
    service_center: Optional[str] = None
    notes: Optional[str] = None
    call_transcript: Optional[str] = None


class AzureSpeechManager:
    """
    Manages Azure Speech SDK for Text-to-Speech and Speech-to-Text operations.
    Handles real-time voice conversion for phone calls.
    """
    
    def __init__(self, speech_key: str, speech_region: str):
        """
        Initialize Azure Speech Manager.
        
        Args:
            speech_key: Azure Speech API key
            speech_region: Azure Speech service region
        """
        self.speech_key = speech_key
        self.speech_region = speech_region
        self.speech_config = None
        
        if not AZURE_SPEECH_AVAILABLE:
            print("Warning: Azure Speech SDK not installed. Voice features disabled.")
            return
        
        # Configure speech
        self.speech_config = speechsdk.SpeechConfig(
            subscription=speech_key,
            region=speech_region
        )
        
        # Set voice for natural conversation
        self.speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
        
        # Configure for phone call quality
        self.speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        )
        
    def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech audio.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio bytes in MP3 format
        """
        if not AZURE_SPEECH_AVAILABLE or not self.speech_config:
            return b""
        
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config,
            audio_config=None  # Output to memory
        )
        
        result = synthesizer.speak_text_async(text).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return result.audio_data
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            raise Exception(f"Speech synthesis canceled: {cancellation.reason}")
        
        return b""
    
    def text_to_speech_ssml(self, text: str, emotion: str = "friendly") -> bytes:
        """
        Convert text to speech with SSML for more natural conversation.
        
        Args:
            text: Text to convert
            emotion: Speaking style (friendly, professional, empathetic)
            
        Returns:
            Audio bytes
        """
        style_map = {
            "friendly": "friendly",
            "professional": "professional",
            "empathetic": "empathetic",
            "cheerful": "cheerful"
        }
        
        style = style_map.get(emotion, "friendly")
        
        ssml = f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" 
               xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
            <voice name="en-US-JennyNeural">
                <mstts:express-as style="{style}">
                    {text}
                </mstts:express-as>
            </voice>
        </speak>
        """
        
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config,
            audio_config=None
        )
        
        result = synthesizer.speak_ssml_async(ssml).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return result.audio_data
        
        return b""
    
    def create_speech_recognizer(self, audio_stream) -> Any:
        """
        Create a speech recognizer for processing incoming audio.
        
        Args:
            audio_stream: Audio input stream
            
        Returns:
            Configured SpeechRecognizer
        """
        if not AZURE_SPEECH_AVAILABLE or not speechsdk:
            return None
            
        audio_config = speechsdk.audio.AudioConfig(stream=audio_stream)
        
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config
        )
        
        return recognizer
    
    async def recognize_continuous(
        self, 
        audio_stream,
        on_recognized: Callable[[str], None],
        timeout_seconds: int = 30
    ):
        """
        Continuously recognize speech from audio stream.
        
        Args:
            audio_stream: Input audio stream
            on_recognized: Callback for recognized text
            timeout_seconds: Recognition timeout
        """
        if not AZURE_SPEECH_AVAILABLE or not speechsdk:
            return
            
        recognizer = self.create_speech_recognizer(audio_stream)
        if not recognizer:
            return
        
        done = asyncio.Event()
        
        def on_recognized_callback(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                on_recognized(evt.result.text)
            elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                pass  # No speech recognized
        
        def on_session_stopped(evt):
            done.set()
        
        recognizer.recognized.connect(on_recognized_callback)
        recognizer.session_stopped.connect(on_session_stopped)
        
        recognizer.start_continuous_recognition()
        
        try:
            await asyncio.wait_for(done.wait(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            pass
        finally:
            recognizer.stop_continuous_recognition()


class VoiceBookingConversationAgent:
    """
    LangChain-powered conversational agent for handling service center calls.
    Manages the dialog flow for booking appointments.
    Uses Google Gemini for AI conversation.
    """
    
    def __init__(self, google_api_key: str, booking_request: BookingRequest):
        """
        Initialize the conversation agent.
        
        Args:
            google_api_key: Google API key for Gemini
            booking_request: Details of the booking to make
        """
        self.booking_request = booking_request
        self.conversation_history = []
        self.booking_confirmed = False
        self.confirmation_details = {}
        self.llm = None
        self.memory = None
        
        # Initialize Gemini LLM if available
        if LANGCHAIN_AVAILABLE and ChatGoogleGenerativeAI:
            self.llm = ChatGoogleGenerativeAI(
                google_api_key=google_api_key,
                model="gemini-1.5-pro",
                temperature=0.7,
                convert_system_message_to_human=True  # Gemini handles system prompts differently
            )
            
            # Memory for conversation
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        else:
            print("Warning: LangChain/Gemini not available. Using simulation mode.")
        
        # System prompt for the booking agent
        self.system_prompt = self._create_system_prompt()
        
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the booking agent."""
        return f"""You are an AI assistant making a phone call to schedule a vehicle service appointment on behalf of a customer. 

CUSTOMER INFORMATION:
- Name: {self.booking_request.customer_name}
- Phone: {self.booking_request.customer_phone}
- Email: {self.booking_request.customer_email}
- Vehicle ID: {self.booking_request.vehicle_id}

VEHICLE ISSUE:
- Issue Type: {self.booking_request.issue_type}
- Description: {self.booking_request.issue_description}
- Severity: {self.booking_request.severity}

PREFERRED APPOINTMENT:
- Date: {self.booking_request.preferred_date.strftime('%A, %B %d, %Y')}
- Time: {self.booking_request.preferred_time}

SERVICE CENTER:
- Name: {self.booking_request.service_center_name}

YOUR TASK:
1. Introduce yourself as calling from VehicleCare AI on behalf of the customer
2. Explain the vehicle issue clearly and professionally
3. Request an appointment for the preferred date/time
4. Be flexible if they suggest alternative times
5. Confirm all booking details before ending the call
6. Get a confirmation number if available

CONVERSATION GUIDELINES:
- Be polite and professional
- Speak clearly and at a moderate pace
- Listen carefully to the service center representative
- Confirm important details by repeating them back
- If they ask questions, answer based on the information provided
- If they mention something you don't know, say you'll have the customer follow up

When the booking is confirmed, include [BOOKING_CONFIRMED] in your response along with:
- Confirmation number (if provided)
- Scheduled date and time
- Any special instructions

If the booking cannot be made, include [BOOKING_FAILED] and explain why.

Respond naturally as if you're on an actual phone call. Keep responses concise and conversational."""

    def get_opening_message(self) -> str:
        """Get the initial message to start the call."""
        return f"""Hello, good morning! My name is Sara and I'm calling from VehicleCare AI on behalf of {self.booking_request.customer_name}. 
I'd like to schedule a service appointment for their vehicle. 
The vehicle has been experiencing {self.booking_request.issue_type.lower()} issues and our diagnostic system has identified it as {self.booking_request.severity.lower()} priority. 
Would you be able to help me book an appointment?"""

    async def process_response(self, service_center_response: str) -> str:
        """
        Process the service center's response and generate the next reply.
        
        Args:
            service_center_response: What the service center representative said
            
        Returns:
            The agent's next response
        """
        # Store response in history (as dict if LangChain not available)
        if LANGCHAIN_AVAILABLE and HumanMessage:
            self.conversation_history.append(
                HumanMessage(content=f"Service Center: {service_center_response}")
            )
        else:
            self.conversation_history.append({
                "role": "service_center",
                "content": service_center_response
            })
        
        # If LLM is available, use it
        if self.llm and LANGCHAIN_AVAILABLE:
            # Create prompt
            messages = [
                SystemMessage(content=self.system_prompt),
                *self.conversation_history
            ]
            
            # Get response from LLM
            response = await self.llm.ainvoke(messages)
            response_content = response.content
            
            # Add AI response to history
            self.conversation_history.append(
                AIMessage(content=response_content)
            )
        else:
            # Simulation mode - generate scripted responses
            response_content = self._get_simulated_response(service_center_response)
            self.conversation_history.append({
                "role": "ai",
                "content": response_content
            })
        
        # Check for booking confirmation
        if "[BOOKING_CONFIRMED]" in response_content:
            self.booking_confirmed = True
            self._extract_confirmation_details(response_content)
        
        # Clean response for speech (remove markers)
        clean_response = response_content.replace("[BOOKING_CONFIRMED]", "").replace("[BOOKING_FAILED]", "")
        
        return clean_response.strip()
    
    def _get_simulated_response(self, service_center_response: str) -> str:
        """Generate simulated AI responses when LLM is not available."""
        response_lower = service_center_response.lower()
        
        # Simple keyword-based responses for simulation
        if "how can i help" in response_lower or "hello" in response_lower:
            return f"Yes, I'm calling on behalf of {self.booking_request.customer_name}. We need to schedule a service appointment for a {self.booking_request.issue_type} issue. The vehicle ID is {self.booking_request.vehicle_id}. Do you have availability on {self.booking_request.preferred_date.strftime('%B %d')} around {self.booking_request.preferred_time}?"
        
        elif "availability" in response_lower or "check" in response_lower or "opening" in response_lower:
            return f"That would be perfect! The customer's contact number is {self.booking_request.customer_phone} and email is {self.booking_request.customer_email}. Can you confirm the booking?"
        
        elif "confirm" in response_lower or "booked" in response_lower or "confirmation" in response_lower:
            self.booking_confirmed = True
            return "[BOOKING_CONFIRMED] Wonderful! Thank you so much for your help. The customer will be there on time. Have a great day!"
        
        elif "name" in response_lower or "contact" in response_lower:
            return f"The customer's name is {self.booking_request.customer_name}, phone number is {self.booking_request.customer_phone}."
        
        else:
            return f"I understand. To clarify, we need a service appointment for {self.booking_request.issue_type}. This is a {self.booking_request.severity} priority issue. Would {self.booking_request.preferred_date.strftime('%B %d')} at {self.booking_request.preferred_time} work?"
    
    def _extract_confirmation_details(self, response: str):
        """Extract booking confirmation details from the response."""
        # This is a simplified extraction - in production, use more robust parsing
        lines = response.lower().split('\n')
        
        for line in lines:
            if 'confirmation' in line and 'number' in line:
                # Extract confirmation number
                parts = line.split(':')
                if len(parts) > 1:
                    self.confirmation_details['confirmation_number'] = parts[1].strip()
            elif 'date' in line:
                parts = line.split(':')
                if len(parts) > 1:
                    self.confirmation_details['scheduled_date'] = parts[1].strip()
            elif 'time' in line:
                parts = line.split(':')
                if len(parts) > 1:
                    self.confirmation_details['scheduled_time'] = parts[1].strip()
    
    def get_conversation_transcript(self) -> str:
        """Get the full conversation transcript."""
        transcript = []
        for msg in self.conversation_history:
            if isinstance(msg, dict):
                # Simulation mode format
                if msg.get("role") == "service_center":
                    transcript.append(f"Service Center: {msg['content']}")
                else:
                    transcript.append(f"VehicleCare AI: {msg['content']}")
            elif LANGCHAIN_AVAILABLE and HumanMessage and isinstance(msg, HumanMessage):
                transcript.append(f"Service Center: {msg.content.replace('Service Center: ', '')}")
            elif LANGCHAIN_AVAILABLE and AIMessage and isinstance(msg, AIMessage):
                transcript.append(f"VehicleCare AI: {msg.content}")
        return "\n\n".join(transcript)
    
    def is_booking_confirmed(self) -> bool:
        """Check if booking has been confirmed."""
        return self.booking_confirmed
    
    def get_confirmation_details(self) -> dict:
        """Get the confirmation details."""
        return self.confirmation_details


class TwilioCallManager:
    """
    Manages Twilio phone calls for automated booking.
    Handles outbound calls and audio streaming.
    """
    
    def __init__(
        self, 
        account_sid: str, 
        auth_token: str, 
        from_number: str,
        webhook_base_url: str
    ):
        """
        Initialize Twilio Call Manager.
        
        Args:
            account_sid: Twilio Account SID
            auth_token: Twilio Auth Token
            from_number: Twilio phone number to call from
            webhook_base_url: Base URL for webhooks (e.g., your ngrok URL)
        """
        from twilio.rest import Client
        from twilio.twiml.voice_response import VoiceResponse, Gather, Say, Stream
        
        self.client = Client(account_sid, auth_token)
        self.from_number = from_number
        self.webhook_base_url = webhook_base_url
        self.active_calls = {}
        
    def initiate_call(self, to_number: str, booking_id: str) -> str:
        """
        Initiate an outbound call to the service center.
        
        Args:
            to_number: Phone number to call
            booking_id: Unique identifier for this booking request
            
        Returns:
            Call SID
        """
        call = self.client.calls.create(
            to=to_number,
            from_=self.from_number,
            url=f"{self.webhook_base_url}/voice/outbound/{booking_id}",
            status_callback=f"{self.webhook_base_url}/voice/status/{booking_id}",
            status_callback_event=['initiated', 'ringing', 'answered', 'completed'],
            record=True,  # Record the call for quality assurance
            machine_detection='Enable'  # Detect if voicemail answers
        )
        
        self.active_calls[booking_id] = call.sid
        return call.sid
    
    def generate_twiml_response(self, text: str, gather: bool = True) -> str:
        """
        Generate TwiML response for the call.
        
        Args:
            text: Text to speak
            gather: Whether to gather response from the other party
            
        Returns:
            TwiML XML string
        """
        from twilio.twiml.voice_response import VoiceResponse, Gather
        
        response = VoiceResponse()
        
        if gather:
            gather = Gather(
                input='speech',
                action=f"{self.webhook_base_url}/voice/gather",
                method='POST',
                speech_timeout='auto',
                language='en-US'
            )
            gather.say(text, voice='Polly.Joanna', language='en-US')
            response.append(gather)
        else:
            response.say(text, voice='Polly.Joanna', language='en-US')
        
        return str(response)
    
    def generate_stream_twiml(self, booking_id: str) -> str:
        """
        Generate TwiML for bidirectional media streaming.
        This enables real-time speech processing.
        
        Args:
            booking_id: Booking identifier
            
        Returns:
            TwiML XML string
        """
        from twilio.twiml.voice_response import VoiceResponse, Start, Stream
        
        response = VoiceResponse()
        
        start = Start()
        stream = Stream(
            url=f"wss://{self.webhook_base_url.replace('https://', '')}/voice/stream/{booking_id}",
            track='both_tracks'
        )
        start.append(stream)
        response.append(start)
        
        # Initial greeting
        response.say(
            "Please hold while I connect you with our automated booking system.",
            voice='Polly.Joanna'
        )
        response.pause(length=60)  # Keep call alive
        
        return str(response)
    
    def end_call(self, booking_id: str):
        """
        End an active call.
        
        Args:
            booking_id: Booking identifier
        """
        if booking_id in self.active_calls:
            call_sid = self.active_calls[booking_id]
            self.client.calls(call_sid).update(status='completed')
            del self.active_calls[booking_id]


class AutomatedBookingSystem:
    """
    Main orchestrator for the automated booking system.
    Coordinates between Speech SDK, LangChain (with Gemini), and Twilio.
    """
    
    def __init__(
        self,
        google_api_key: str,
        azure_speech_key: str,
        azure_speech_region: str,
        twilio_account_sid: Optional[str] = None,
        twilio_auth_token: Optional[str] = None,
        twilio_phone_number: Optional[str] = None,
        webhook_base_url: Optional[str] = None
    ):
        """
        Initialize the Automated Booking System.
        
        Args:
            google_api_key: Google API key for Gemini
            azure_speech_key: Azure Speech SDK key
            azure_speech_region: Azure Speech service region
            twilio_account_sid: Twilio account SID (optional for simulation mode)
            twilio_auth_token: Twilio auth token (optional for simulation mode)
            twilio_phone_number: Twilio phone number (optional for simulation mode)
            webhook_base_url: Webhook base URL (optional for simulation mode)
        """
        self.google_api_key = google_api_key
        
        # Initialize Azure Speech
        self.speech_manager = AzureSpeechManager(
            azure_speech_key,
            azure_speech_region
        )
        
        # Initialize Twilio if credentials provided
        self.twilio_manager = None
        if all([twilio_account_sid, twilio_auth_token, twilio_phone_number, webhook_base_url]):
            self.twilio_manager = TwilioCallManager(
                twilio_account_sid,
                twilio_auth_token,
                twilio_phone_number,
                webhook_base_url
            )
        
        # Active booking sessions
        self.active_sessions = {}
        
    async def start_booking_call(
        self, 
        booking_request: BookingRequest,
        status_callback: Optional[Callable[[BookingStatus, str], None]] = None
    ) -> BookingResult:
        """
        Start an automated booking call.
        
        Args:
            booking_request: Details of the booking
            status_callback: Optional callback for status updates
            
        Returns:
            BookingResult with the outcome
        """
        booking_id = f"booking_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create conversation agent
        agent = VoiceBookingConversationAgent(
            self.google_api_key,
            booking_request
        )
        
        self.active_sessions[booking_id] = {
            'agent': agent,
            'request': booking_request,
            'status': BookingStatus.PENDING
        }
        
        if status_callback:
            status_callback(BookingStatus.CALLING, "Initiating call to service center...")
        
        # If Twilio is configured, make a real call
        if self.twilio_manager:
            try:
                call_sid = self.twilio_manager.initiate_call(
                    booking_request.service_center_phone,
                    booking_id
                )
                
                self.active_sessions[booking_id]['call_sid'] = call_sid
                self.active_sessions[booking_id]['status'] = BookingStatus.IN_PROGRESS
                
                if status_callback:
                    status_callback(
                        BookingStatus.IN_PROGRESS, 
                        f"Call connected. Speaking with {booking_request.service_center_name}..."
                    )
                
                # Wait for call to complete (handled by webhooks in production)
                # For now, return a pending result
                return BookingResult(
                    status=BookingStatus.IN_PROGRESS,
                    notes=f"Call initiated. Call SID: {call_sid}"
                )
                
            except Exception as e:
                return BookingResult(
                    status=BookingStatus.FAILED,
                    notes=f"Failed to initiate call: {str(e)}"
                )
        else:
            # Simulation mode - simulate the conversation
            return await self._simulate_booking_call(booking_id, agent, status_callback)
    
    async def _simulate_booking_call(
        self,
        booking_id: str,
        agent: VoiceBookingConversationAgent,
        status_callback: Optional[Callable[[BookingStatus, str], None]] = None
    ) -> BookingResult:
        """
        Simulate a booking call for testing/demo purposes.
        """
        if status_callback:
            status_callback(BookingStatus.IN_PROGRESS, "Connecting to service center (simulation mode)...")
        
        # Simulated service center responses
        simulated_responses = [
            "Hello, thank you for calling VehicleCare Certified Center. How can I help you today?",
            "I understand. Let me check our availability for that date. One moment please.",
            f"We have an opening on {agent.booking_request.preferred_date.strftime('%B %d')} at {agent.booking_request.preferred_time}. Would that work?",
            "Perfect! I'll book that appointment for you. Can you confirm the vehicle owner's name and contact number?",
            f"Great, I have {agent.booking_request.customer_name} at {agent.booking_request.customer_phone}. Your confirmation number is VC{datetime.now().strftime('%Y%m%d%H%M')}. Is there anything else I can help you with?",
            "Thank you for choosing VehicleCare. Have a great day!"
        ]
        
        # Get opening message
        opening = agent.get_opening_message()
        
        # Generate speech for opening (demonstrates TTS capability)
        try:
            audio_data = self.speech_manager.text_to_speech(opening)
            if status_callback:
                status_callback(BookingStatus.IN_PROGRESS, f"AI: {opening[:100]}...")
        except Exception as e:
            if status_callback:
                status_callback(BookingStatus.IN_PROGRESS, f"[TTS not available - text mode]: {opening[:100]}...")
        
        # Simulate conversation
        for response in simulated_responses:
            await asyncio.sleep(1)  # Simulate conversation delay
            
            if status_callback:
                status_callback(BookingStatus.IN_PROGRESS, f"Service Center: {response[:80]}...")
            
            # Get agent's response
            agent_reply = await agent.process_response(response)
            
            if status_callback:
                status_callback(BookingStatus.IN_PROGRESS, f"AI: {agent_reply[:80]}...")
            
            await asyncio.sleep(0.5)
            
            # Check if booking is confirmed
            if agent.is_booking_confirmed():
                break
        
        # Get final result
        if agent.is_booking_confirmed():
            details = agent.get_confirmation_details()
            return BookingResult(
                status=BookingStatus.CONFIRMED,
                confirmation_number=details.get('confirmation_number', f"VC{datetime.now().strftime('%Y%m%d%H%M')}"),
                scheduled_date=agent.booking_request.preferred_date.strftime('%A, %B %d, %Y'),
                scheduled_time=agent.booking_request.preferred_time,
                service_center=agent.booking_request.service_center_name,
                call_transcript=agent.get_conversation_transcript(),
                notes="Appointment successfully booked via automated call."
            )
        else:
            return BookingResult(
                status=BookingStatus.CONFIRMED,  # In simulation, always confirm
                confirmation_number=f"VC{datetime.now().strftime('%Y%m%d%H%M')}",
                scheduled_date=agent.booking_request.preferred_date.strftime('%A, %B %d, %Y'),
                scheduled_time=agent.booking_request.preferred_time,
                service_center=agent.booking_request.service_center_name,
                call_transcript=agent.get_conversation_transcript(),
                notes="Appointment booked via automated call (simulation mode)."
            )
    
    def get_session_status(self, booking_id: str) -> Optional[BookingStatus]:
        """Get the status of a booking session."""
        if booking_id in self.active_sessions:
            return self.active_sessions[booking_id]['status']
        return None
    
    async def handle_incoming_audio(self, booking_id: str, audio_data: bytes) -> Optional[bytes]:
        """
        Handle incoming audio from the call and generate response.
        Used for real-time call handling.
        
        Args:
            booking_id: Booking session identifier
            audio_data: Incoming audio bytes
            
        Returns:
            Response audio bytes
        """
        if booking_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[booking_id]
        agent = session['agent']
        
        # In production, you would:
        # 1. Convert incoming audio to text using Speech-to-Text
        # 2. Process with the conversation agent
        # 3. Convert response to audio using Text-to-Speech
        # 4. Return the audio bytes
        
        # This is a placeholder for the WebSocket-based real-time handling
        return None


# Convenience function for Streamlit integration
async def book_appointment_automatically(
    booking_request: BookingRequest,
    google_api_key: str,
    azure_speech_key: str,
    azure_speech_region: str,
    status_callback: Optional[Callable[[BookingStatus, str], None]] = None,
    twilio_config: Optional[Dict] = None
) -> BookingResult:
    """
    Convenience function to book an appointment automatically.
    
    Args:
        booking_request: Details of the booking
        google_api_key: Google API key for Gemini
        azure_speech_key: Azure Speech SDK key
        azure_speech_region: Azure Speech region
        status_callback: Optional callback for status updates
        twilio_config: Optional Twilio configuration dict
        
    Returns:
        BookingResult with the outcome
    """
    twilio_args = {}
    if twilio_config:
        twilio_args = {
            'twilio_account_sid': twilio_config.get('account_sid'),
            'twilio_auth_token': twilio_config.get('auth_token'),
            'twilio_phone_number': twilio_config.get('phone_number'),
            'webhook_base_url': twilio_config.get('webhook_base_url')
        }
    
    system = AutomatedBookingSystem(
        google_api_key=google_api_key,
        azure_speech_key=azure_speech_key,
        azure_speech_region=azure_speech_region,
        **twilio_args
    )
    
    return await system.start_booking_call(booking_request, status_callback)


# Service center phone directory (simulated)
SERVICE_CENTER_DIRECTORY = {
    "VehicleCare Certified Center - Downtown": {
        "phone": "+1-555-0101",
        "address": "123 Main Street, Downtown",
        "hours": "8:00 AM - 6:00 PM"
    },
    "VehicleCare Certified Center - Uptown": {
        "phone": "+1-555-0102",
        "address": "456 Oak Avenue, Uptown",
        "hours": "8:00 AM - 6:00 PM"
    },
    "VehicleCare Certified Center - Westside": {
        "phone": "+1-555-0103",
        "address": "789 West Boulevard, Westside",
        "hours": "8:00 AM - 6:00 PM"
    },
    "VehicleCare Certified Center - Eastside": {
        "phone": "+1-555-0104",
        "address": "321 East Drive, Eastside",
        "hours": "8:00 AM - 6:00 PM"
    }
}


def get_service_center_phone(center_name: str) -> str:
    """Get the phone number for a service center."""
    center = SERVICE_CENTER_DIRECTORY.get(center_name, {})
    return center.get("phone", "+1-555-0100")


@dataclass
class AutoBookingProgress:
    """Progress update for auto-booking process."""
    current_center: str
    center_index: int
    total_centers: int
    status: str  # "calling", "no_answer", "busy", "confirmed", "failed"
    message: str
    result: Optional[BookingResult] = None


async def auto_book_with_service_centers(
    customer_name: str,
    customer_phone: str,
    customer_email: str,
    vehicle_id: str,
    issue_type: str,
    issue_description: str,
    severity: str,
    google_api_key: str,
    azure_speech_key: str = "",
    azure_speech_region: str = "eastus",
    progress_callback: Optional[Callable[[AutoBookingProgress], None]] = None,
    twilio_config: Optional[Dict] = None
) -> BookingResult:
    """
    Automatically book with service centers by calling them one by one.
    Stops as soon as one confirms the booking.
    
    In simulation mode (no Twilio), simulates calling centers with
    realistic delays and confirms with a random center.
    """
    import random
    
    service_centers = list(SERVICE_CENTER_DIRECTORY.keys())
    total_centers = len(service_centers)
    
    # Calculate preferred date/time (next business day, 10 AM)
    now = datetime.now()
    preferred_date = now + timedelta(days=1)
    # Skip to Monday if weekend
    while preferred_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
        preferred_date += timedelta(days=1)
    preferred_time = "10:00 AM"
    
    # In simulation mode, randomly pick which center will confirm (usually 2nd or 3rd)
    # This simulates real-world scenario where first few might be busy
    confirm_at_index = random.randint(1, min(2, total_centers - 1))
    
    # Try each service center
    for idx, center_name in enumerate(service_centers):
        center_info = SERVICE_CENTER_DIRECTORY[center_name]
        
        # Report progress - calling this center
        if progress_callback:
            progress_callback(AutoBookingProgress(
                current_center=center_name,
                center_index=idx + 1,
                total_centers=total_centers,
                status="calling",
                message=f"📞 Calling {center_name}..."
            ))
        
        # Simulate call duration (1-2 seconds per center)
        await asyncio.sleep(random.uniform(1.0, 2.0))
        
        # Check if this is the center that confirms (simulation)
        # Or if we have real Twilio config, make actual call
        if twilio_config and all(twilio_config.values()):
            # Real Twilio call - use full booking flow
            booking_request = BookingRequest(
                customer_name=customer_name,
                customer_phone=customer_phone,
                customer_email=customer_email,
                vehicle_id=vehicle_id,
                issue_type=issue_type,
                issue_description=issue_description,
                severity=severity,
                preferred_date=preferred_date,
                preferred_time=preferred_time,
                service_center_phone=center_info["phone"],
                service_center_name=center_name
            )
            
            try:
                result = await book_appointment_automatically(
                    booking_request=booking_request,
                    google_api_key=google_api_key,
                    azure_speech_key=azure_speech_key,
                    azure_speech_region=azure_speech_region,
                    status_callback=None,
                    twilio_config=twilio_config
                )
                
                if result.status == BookingStatus.CONFIRMED:
                    if progress_callback:
                        progress_callback(AutoBookingProgress(
                            current_center=center_name,
                            center_index=idx + 1,
                            total_centers=total_centers,
                            status="confirmed",
                            message=f"✅ Booking confirmed with {center_name}!",
                            result=result
                        ))
                    return result
            except Exception:
                pass  # Try next center
        else:
            # Simulation mode - confirm at designated index
            if idx == confirm_at_index:
                # This center confirms!
                confirmation_number = f"VC{datetime.now().strftime('%Y%m%d%H%M')}"
                
                # Generate simulated conversation transcript
                transcript = f"""VehicleCare AI: Hello, I'm calling from VehicleCare AI on behalf of {customer_name}. We need to schedule a service appointment for a {issue_type} issue.

Service Center: Hello! Thank you for calling {center_name}. I'd be happy to help you schedule that appointment.

VehicleCare AI: Great! The vehicle ID is {vehicle_id} and this is a {severity} priority issue. Do you have availability on {preferred_date.strftime('%B %d')} around {preferred_time}?

Service Center: Let me check... Yes, we have an opening at {preferred_time} on {preferred_date.strftime('%B %d')}. 

VehicleCare AI: Perfect! The customer's contact number is {customer_phone} and email is {customer_email}.

Service Center: I've booked the appointment. Your confirmation number is {confirmation_number}. We'll see you on {preferred_date.strftime('%B %d')} at {preferred_time}.

VehicleCare AI: Thank you so much! Have a great day!"""

                result = BookingResult(
                    status=BookingStatus.CONFIRMED,
                    confirmation_number=confirmation_number,
                    scheduled_date=preferred_date.strftime('%A, %B %d, %Y'),
                    scheduled_time=preferred_time,
                    service_center=center_name,
                    call_transcript=transcript,
                    notes=f"Appointment booked via automated AI call to {center_name}."
                )
                
                if progress_callback:
                    progress_callback(AutoBookingProgress(
                        current_center=center_name,
                        center_index=idx + 1,
                        total_centers=total_centers,
                        status="confirmed",
                        message=f"✅ Booking confirmed with {center_name}!",
                        result=result
                    ))
                
                return result
        
        # This center didn't confirm - report and try next
        if progress_callback:
            progress_callback(AutoBookingProgress(
                current_center=center_name,
                center_index=idx + 1,
                total_centers=total_centers,
                status="no_answer",
                message=f"❌ {center_name} - No availability"
            ))
        
        # Small delay before next call
        await asyncio.sleep(0.3)
    
    # All centers tried, none confirmed (shouldn't happen in simulation)
    return BookingResult(
        status=BookingStatus.FAILED,
        notes="Unable to book with any service center. Please try manual booking."
    )


def run_auto_booking_sync(
    customer_name: str,
    customer_phone: str,
    customer_email: str,
    vehicle_id: str,
    issue_type: str,
    issue_description: str,
    severity: str,
    google_api_key: str,
    azure_speech_key: str = "",
    azure_speech_region: str = "eastus",
    progress_callback: Optional[Callable[[AutoBookingProgress], None]] = None
) -> BookingResult:
    """
    Synchronous wrapper for auto_book_with_service_centers.
    Use this from Streamlit.
    """
    return asyncio.run(auto_book_with_service_centers(
        customer_name=customer_name,
        customer_phone=customer_phone,
        customer_email=customer_email,
        vehicle_id=vehicle_id,
        issue_type=issue_type,
        issue_description=issue_description,
        severity=severity,
        google_api_key=google_api_key,
        azure_speech_key=azure_speech_key,
        azure_speech_region=azure_speech_region,
        progress_callback=progress_callback
    ))

