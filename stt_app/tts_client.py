"""
Microsoft Edge Text-to-Speech Client (Free)
Handles audio generation from translated text using edge-tts
"""

import logging
import asyncio
import tempfile
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TTSClient:
    """Client for Microsoft Edge Text-to-Speech (free)."""
    
    def __init__(self) -> None:
        # No API key needed - edge-tts is completely free!
        self._pygame = None
    
    def _ensure_playback_backend(self):
        if self._pygame is None:
            import pygame

            pygame.mixer.init()
            self._pygame = pygame
        return self._pygame
    
    def is_configured(self) -> bool:
        """Always configured - no API key needed."""
        return True
    
    async def _text_to_speech_async(
        self, 
        text: str, 
        voice: str,
        output_file: str
    ) -> None:
        """
        Convert text to speech asynchronously and save to file.
        
        Args:
            text: Text to convert to speech
            voice: Voice ID (e.g., 'de-DE-KatjaNeural')
            output_file: Path to save MP3 file
        """
        import edge_tts

        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)
    
    def text_to_speech_and_play(
        self, 
        text: str, 
        voice: str = "de-DE-KatjaNeural"
    ) -> None:
        """
        Convert text to speech and play it immediately.
        
        Args:
            text: Text to convert to speech
            voice: Voice ID (default: German female)
            
        Raises:
            Exception: If TTS conversion or playback fails
        """
        try:
            logger.info(f"Generating speech for text: {text[:50]}... (voice: {voice})")
            
            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # Generate speech asynchronously
            asyncio.run(self._text_to_speech_async(text, voice, tmp_path))
            
            # Play the audio file using pygame (Windows-compatible)
            pygame = self._ensure_playback_backend()
            logger.info(f"Playing audio from {tmp_path}")
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # Clean up temporary file
            try:
                Path(tmp_path).unlink()
            except Exception as e:
                logger.warning(f"Could not delete temp audio file: {e}")
                
        except Exception as e:
            logger.error(f"TTS conversion or playback failed: {e}")
            raise


# Voice options for different languages (Microsoft Edge voices)
# These are high-quality neural voices, completely free!
VOICE_OPTIONS = {
    "Deutsch": "de-DE-KatjaNeural",      # German female (clear, natural)
    "Englisch": "en-US-AriaNeural",      # English female (clear, professional)
    "Spanisch": "es-ES-ElviraNeural",    # Spanish female (clear)
    "Französisch": "fr-FR-DeniseNeural", # French female (clear)
    "Italienisch": "it-IT-ElsaNeural",   # Italian female (clear)
    "Arabisch": "ar-SA-ZariyahNeural",   # Arabic female (clear)
}


# Alternative male voices (if needed)
VOICE_OPTIONS_MALE = {
    "Deutsch": "de-DE-ConradNeural",
    "Englisch": "en-US-GuyNeural",
    "Spanisch": "es-ES-AlvaroNeural",
    "Französisch": "fr-FR-HenriNeural",
    "Italienisch": "it-IT-DiegoNeural",
    "Arabisch": "ar-SA-HamedNeural",
}
