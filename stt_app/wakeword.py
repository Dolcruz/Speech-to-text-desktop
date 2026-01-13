"""Wake word detection using OpenWakeWord.

Continuously listens for wake words and triggers callbacks when detected.
Runs in a background thread to avoid blocking the UI.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional, List

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

# Available wake word models from OpenWakeWord
AVAILABLE_WAKE_WORDS = {
    "hey_jarvis": "Hey Jarvis",
    "alexa": "Alexa",
    "hey_mycroft": "Hey Mycroft",
    "hey_rhasspy": "Hey Rhasspy",
}

# Default threshold for wake word detection (0-1)
DEFAULT_THRESHOLD = 0.5


class WakeWordDetector:
    """Background wake word detection using OpenWakeWord.

    Listens continuously for configured wake words and invokes a callback
    when one is detected with sufficient confidence.
    """

    def __init__(
        self,
        wake_word: str = "hey_jarvis",
        threshold: float = DEFAULT_THRESHOLD,
        on_detected: Optional[Callable[[], None]] = None,
        sample_rate: int = 16000,
        input_device_index: Optional[int] = None,
    ) -> None:
        """Initialize the wake word detector.

        Args:
            wake_word: Which wake word to listen for (key from AVAILABLE_WAKE_WORDS)
            threshold: Confidence threshold (0-1) for triggering detection
            on_detected: Callback invoked when wake word is detected
            sample_rate: Audio sample rate (must be 16000 for OpenWakeWord)
            input_device_index: Optional specific input device to use
        """
        self.wake_word = wake_word
        self.threshold = threshold
        self.on_detected = on_detected
        self.sample_rate = sample_rate
        self.input_device_index = input_device_index

        self._model = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._paused.set()  # Start in non-paused state
        self._lock = threading.Lock()

        # Cooldown to prevent multiple triggers
        self._last_detection_time: float = 0.0
        self._cooldown_seconds: float = 2.0

        # Model loading state
        self._model_loaded = False
        self._model_error: Optional[str] = None

    def _ensure_model(self) -> bool:
        """Lazy-load the OpenWakeWord model.

        Returns:
            True if model is ready, False if loading failed.
        """
        if self._model is not None:
            return True

        try:
            import openwakeword
            from openwakeword.model import Model

            # Download models if not present
            logger.info("Downloading OpenWakeWord models if needed...")
            openwakeword.utils.download_models()

            # Load the model
            logger.info("Loading wake word model: %s", self.wake_word)
            self._model = Model(
                wakeword_models=[self.wake_word],
                inference_framework="onnx"  # Use ONNX on Windows
            )
            self._model_loaded = True
            logger.info("Wake word model loaded successfully")
            return True

        except ImportError as e:
            self._model_error = f"OpenWakeWord not installed: {e}"
            logger.error(self._model_error)
            return False
        except Exception as e:
            self._model_error = f"Failed to load wake word model: {e}"
            logger.error(self._model_error)
            return False

    def is_running(self) -> bool:
        """Check if the detector is currently running."""
        return self._thread is not None and self._thread.is_alive()

    def is_paused(self) -> bool:
        """Check if the detector is paused."""
        return not self._paused.is_set()

    def start(self) -> bool:
        """Start listening for wake words in a background thread.

        Returns:
            True if started successfully, False otherwise.
        """
        if self.is_running():
            logger.warning("Wake word detector already running")
            return True

        # Load model first
        if not self._ensure_model():
            return False

        self._stop_event.clear()
        self._paused.set()  # Start in non-paused state
        self._thread = threading.Thread(
            target=self._run,
            name="WakeWordThread",
            daemon=True
        )
        self._thread.start()
        logger.info("Wake word detector started")
        return True

    def stop(self) -> None:
        """Stop the wake word detector."""
        self._stop_event.set()
        self._paused.set()  # Unpause to allow thread to exit
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Wake word detector stopped")

    def pause(self) -> None:
        """Temporarily pause wake word detection (e.g., during recording)."""
        self._paused.clear()
        logger.debug("Wake word detection paused")

    def resume(self) -> None:
        """Resume wake word detection after pause."""
        # Add small cooldown after resume to avoid immediate re-trigger
        self._last_detection_time = time.time()
        self._paused.set()
        logger.debug("Wake word detection resumed")

    def set_wake_word(self, wake_word: str) -> None:
        """Change the wake word (requires restart to take effect)."""
        self.wake_word = wake_word
        self._model = None  # Force model reload
        self._model_loaded = False

    def set_threshold(self, threshold: float) -> None:
        """Update the detection threshold."""
        self.threshold = max(0.0, min(1.0, threshold))

    def _run(self) -> None:
        """Main detection loop running in background thread."""
        # Audio settings for OpenWakeWord
        chunk_size = int(self.sample_rate * 0.08)  # 80ms chunks

        try:
            # Set up audio stream
            stream_kwargs = {
                "channels": 1,
                "samplerate": self.sample_rate,
                "dtype": "int16",
                "blocksize": chunk_size,
            }
            if self.input_device_index is not None:
                stream_kwargs["device"] = self.input_device_index

            with sd.InputStream(**stream_kwargs) as stream:
                logger.info("Wake word audio stream opened")

                while not self._stop_event.is_set():
                    # Wait if paused
                    if not self._paused.wait(timeout=0.1):
                        continue

                    # Check stop again after wait
                    if self._stop_event.is_set():
                        break

                    # Read audio chunk
                    try:
                        audio_data, overflowed = stream.read(chunk_size)
                        if overflowed:
                            logger.debug("Audio buffer overflow")
                            continue
                    except Exception as e:
                        logger.warning("Error reading audio: %s", e)
                        time.sleep(0.1)
                        continue

                    # Convert to format expected by OpenWakeWord
                    audio_array = audio_data.flatten()

                    # Run prediction
                    try:
                        prediction = self._model.predict(audio_array)

                        # Check if wake word detected
                        for model_name, score in prediction.items():
                            if score >= self.threshold:
                                # Check cooldown
                                now = time.time()
                                if now - self._last_detection_time >= self._cooldown_seconds:
                                    self._last_detection_time = now
                                    logger.info(
                                        "Wake word '%s' detected with confidence %.2f",
                                        model_name, score
                                    )
                                    if self.on_detected:
                                        try:
                                            self.on_detected()
                                        except Exception as e:
                                            logger.error("Error in detection callback: %s", e)

                    except Exception as e:
                        logger.warning("Error during wake word prediction: %s", e)

        except Exception as e:
            logger.error("Wake word detector error: %s", e)
        finally:
            logger.info("Wake word detection loop ended")

    def get_model_error(self) -> Optional[str]:
        """Return the last model loading error, if any."""
        return self._model_error
