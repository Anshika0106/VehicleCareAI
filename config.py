"""
VehicleCare AI - Configuration Management
Handles API keys and settings for the voice booking system.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class VoiceBookingConfig:
    """Configuration for the voice booking system."""
    
    # Google Gemini Configuration
    google_api_key: str = ""
    
    # Azure Speech SDK Configuration
    azure_speech_key: str = ""
    azure_speech_region: str = "eastus"
    
    # Twilio Configuration
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""
    
    # Webhook Configuration
    webhook_base_url: str = ""
    
    # Environment
    environment: str = "development"
    enable_call_recording: bool = True
    
    @classmethod
    def from_env(cls) -> "VoiceBookingConfig":
        """Load configuration from environment variables."""
        return cls(
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            azure_speech_key=os.getenv("AZURE_SPEECH_KEY", ""),
            azure_speech_region=os.getenv("AZURE_SPEECH_REGION", "eastus"),
            twilio_account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
            twilio_auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
            twilio_phone_number=os.getenv("TWILIO_PHONE_NUMBER", ""),
            webhook_base_url=os.getenv("WEBHOOK_BASE_URL", ""),
            environment=os.getenv("ENVIRONMENT", "development"),
            enable_call_recording=os.getenv("ENABLE_CALL_RECORDING", "true").lower() == "true"
        )
    
    @classmethod
    def from_streamlit_secrets(cls) -> "VoiceBookingConfig":
        """Load configuration from Streamlit secrets (for deployment)."""
        try:
            import streamlit as st
            return cls(
                google_api_key=st.secrets.get("GOOGLE_API_KEY", ""),
                azure_speech_key=st.secrets.get("AZURE_SPEECH_KEY", ""),
                azure_speech_region=st.secrets.get("AZURE_SPEECH_REGION", "eastus"),
                twilio_account_sid=st.secrets.get("TWILIO_ACCOUNT_SID", ""),
                twilio_auth_token=st.secrets.get("TWILIO_AUTH_TOKEN", ""),
                twilio_phone_number=st.secrets.get("TWILIO_PHONE_NUMBER", ""),
                webhook_base_url=st.secrets.get("WEBHOOK_BASE_URL", ""),
                environment=st.secrets.get("ENVIRONMENT", "development"),
                enable_call_recording=st.secrets.get("ENABLE_CALL_RECORDING", True)
            )
        except Exception:
            return cls.from_env()
    
    def is_gemini_configured(self) -> bool:
        """Check if Google Gemini is configured."""
        return bool(self.google_api_key and self.google_api_key != "demo-key")
    
    def is_azure_speech_configured(self) -> bool:
        """Check if Azure Speech is configured."""
        return bool(self.azure_speech_key and self.azure_speech_key != "demo-key")
    
    def is_twilio_configured(self) -> bool:
        """Check if Twilio is configured for live calls."""
        return all([
            self.twilio_account_sid,
            self.twilio_auth_token,
            self.twilio_phone_number,
            self.webhook_base_url
        ])
    
    def is_fully_configured(self) -> bool:
        """Check if all services are configured for production use."""
        return (
            self.is_gemini_configured() and
            self.is_azure_speech_configured() and
            self.is_twilio_configured()
        )
    
    def get_configuration_status(self) -> dict:
        """Get detailed configuration status."""
        return {
            "gemini": {
                "configured": self.is_gemini_configured(),
                "status": "✓ Configured" if self.is_gemini_configured() else "✗ Not configured"
            },
            "azure_speech": {
                "configured": self.is_azure_speech_configured(),
                "region": self.azure_speech_region,
                "status": "✓ Configured" if self.is_azure_speech_configured() else "✗ Not configured"
            },
            "twilio": {
                "configured": self.is_twilio_configured(),
                "phone": self.twilio_phone_number if self.is_twilio_configured() else "Not set",
                "status": "✓ Configured" if self.is_twilio_configured() else "✗ Not configured"
            },
            "environment": self.environment,
            "ready_for_live_calls": self.is_fully_configured()
        }


def load_config() -> VoiceBookingConfig:
    """
    Load configuration from the best available source.
    Priority: Streamlit secrets > Environment variables > Defaults
    """
    # Try to load from dotenv if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # First try Streamlit secrets, fall back to env vars
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and st.secrets:
            return VoiceBookingConfig.from_streamlit_secrets()
    except Exception:
        pass
    
    return VoiceBookingConfig.from_env()


# Global config instance
_config: Optional[VoiceBookingConfig] = None


def get_config() -> VoiceBookingConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


# ============================================
# Configuration Setup Instructions
# ============================================
SETUP_INSTRUCTIONS = """
# VehicleCare AI - Voice Booking Setup

## Required API Keys

### 1. Google Gemini API Key (Required for AI Conversation)
   - Go to: https://aistudio.google.com/app/apikey
   - Create an API key
   - Set environment variable: GOOGLE_API_KEY=your-gemini-api-key

### 2. Azure Speech SDK (Required for Voice)
   - Create an Azure account: https://azure.microsoft.com/
   - Create a Speech resource: https://portal.azure.com/#create/Microsoft.CognitiveServicesSpeechServices
   - Get your key and region from the resource
   - Set environment variables:
     - AZURE_SPEECH_KEY=your-key-here
     - AZURE_SPEECH_REGION=eastus (or your region)

### 3. Twilio (Required for Live Phone Calls)
   - Sign up at: https://www.twilio.com/
   - Get your credentials from: https://console.twilio.com/
   - Purchase a phone number capable of voice calls
   - Set environment variables:
     - TWILIO_ACCOUNT_SID=your-account-sid
     - TWILIO_AUTH_TOKEN=your-auth-token
     - TWILIO_PHONE_NUMBER=+15551234567

### 4. Webhook Server (Required for Live Calls)
   - For local development, install ngrok: https://ngrok.com/
   - Run: ngrok http 8000
   - Copy the HTTPS URL
   - Set environment variable: WEBHOOK_BASE_URL=https://your-url.ngrok.io

## Quick Start (Simulation Mode)

You can test the AI booking system in simulation mode without configuring all APIs.
The system will simulate the phone call conversation using the AI agent.

Set minimal configuration:
   - GOOGLE_API_KEY=your-gemini-api-key

## Running the Webhook Server

For live calls, you need to run the webhook server:

```bash
# Terminal 1: Start the webhook server
python twilio_webhook_server.py

# Terminal 2: Start ngrok
ngrok http 8000

# Terminal 3: Start the Streamlit app
streamlit run app.py
```

## Environment Variables

Create a .env file in the project root with:

```
GOOGLE_API_KEY=your-gemini-api-key
AZURE_SPEECH_KEY=your-azure-key
AZURE_SPEECH_REGION=eastus
TWILIO_ACCOUNT_SID=your-sid
TWILIO_AUTH_TOKEN=your-token
TWILIO_PHONE_NUMBER=+15551234567
WEBHOOK_BASE_URL=https://your-ngrok-url.ngrok.io
```
"""


def print_setup_instructions():
    """Print setup instructions to console."""
    print(SETUP_INSTRUCTIONS)


if __name__ == "__main__":
    # Print setup instructions when run directly
    print_setup_instructions()
    
    # Show current configuration status
    config = load_config()
    print("\n" + "=" * 50)
    print("Current Configuration Status:")
    print("=" * 50)
    
    status = config.get_configuration_status()
    for service, info in status.items():
        if isinstance(info, dict):
            print(f"\n{service.upper()}:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print(f"{service}: {info}")

