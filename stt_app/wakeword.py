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
from scipy import signal

from .config import get_app_dir

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
        on_score_update: Optional[Callable[[str, float, float, str], None]] = None,  # (model, score, audio_level, device_name)
        sample_rate: int = 16000,
        input_device_index: Optional[int] = None,
    ) -> None:
        """Initialize the wake word detector.

        Args:
            wake_word: Which wake word to listen for (key from AVAILABLE_WAKE_WORDS)
            threshold: Confidence threshold (0-1) for triggering detection
            on_detected: Callback invoked when wake word is detected
            on_score_update: Callback for live score updates (model_name, score)
            sample_rate: Audio sample rate (must be 16000 for OpenWakeWord)
            input_device_index: Optional specific input device to use
        """
        self.wake_word = wake_word
        self.threshold = threshold
        self.on_detected = on_detected
        self.on_score_update = on_score_update
        self.sample_rate = sample_rate
        self.input_device_index = input_device_index

        # Get device name immediately
        try:
            if input_device_index is not None:
                device_info = sd.query_devices(input_device_index)
                self.device_name = device_info.get('name', f'Device {input_device_index}')
            else:
                device_info = sd.query_devices(kind='input')
                self.device_name = device_info.get('name', 'Default')
        except Exception as e:
            self.device_name = f"Error: {e}"

        logger.info("WakeWordDetector created with device_index=%s, device_name=%s", input_device_index, self.device_name)

        # Debug: track max score for periodic logging
        self._max_score_seen = 0.0
        self._last_debug_log = 0.0

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

    def _download_model_if_missing(self, url: str, dest: Path) -> None:
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading wake word model file: %s", dest.name)
        try:
            import requests
            with requests.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                with open(dest, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            handle.write(chunk)
        except Exception as exc:
            raise RuntimeError(f"Failed to download {url}: {exc}") from exc

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

            model_dir = get_app_dir() / "wakeword_models"
            inference_framework = "onnx"

            def to_model_url(url: str) -> str:
                if inference_framework == "onnx":
                    return url.replace(".tflite", ".onnx")
                return url

            wake_model = openwakeword.MODELS.get(self.wake_word)
            if not wake_model:
                raise ValueError(f"Unknown wake word model: {self.wake_word}")

            wake_url = to_model_url(wake_model["download_url"])
            melspec_url = to_model_url(openwakeword.FEATURE_MODELS["melspectrogram"]["download_url"])
            embed_url = to_model_url(openwakeword.FEATURE_MODELS["embedding"]["download_url"])

            wake_path = model_dir / Path(wake_url).name
            melspec_path = model_dir / Path(melspec_url).name
            embed_path = model_dir / Path(embed_url).name

            logger.info("Ensuring wake word models in %s", model_dir)
            self._download_model_if_missing(melspec_url, melspec_path)
            self._download_model_if_missing(embed_url, embed_path)
            self._download_model_if_missing(wake_url, wake_path)

            # Load the model
            logger.info("Loading wake word model: %s", self.wake_word)
            self._model = Model(
                wakeword_models=[str(wake_path)],
                inference_framework=inference_framework,
                melspec_model_path=str(melspec_path),
                embedding_model_path=str(embed_path),
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
        # OpenWakeWord requires 16kHz; prefer 16kHz input, fall back to native and resample.
        target_rate = 16000  # Required by OpenWakeWord
        native_rate = 44100  # Common microphone rate (HyperX QuadCast etc.)

        # Determine which device to use
        device_to_use = self.input_device_index
        logger.info("Wake word using device index: %s", device_to_use)

        # Try to get actual device sample rate
        try:
            if device_to_use is not None:
                device_info = sd.query_devices(device_to_use)
                self.device_name = device_info.get('name', 'Unknown')
            else:
                device_info = sd.query_devices(kind='input')
                self.device_name = device_info.get('name', 'Default')
            logger.info("Using device: %s", self.device_name)
            native_rate = int(device_info['default_samplerate'])
            logger.info("Device sample rate: %d Hz", native_rate)
        except Exception as e:
            logger.warning("Could not query device sample rate, using %d: %s", native_rate, e)
            self.device_name = "Error"

        # Prefer 16kHz capture to avoid resampling; fall back to native if unsupported.
        stream_rate = target_rate
        try:
            sd.check_input_settings(
                device=device_to_use,
                samplerate=target_rate,
                channels=1,
                dtype="float32",
            )
            logger.info("Using 16kHz input stream for wake word detection")
        except Exception as e:
            stream_rate = native_rate
            logger.info(
                "16kHz input not supported, using native %d Hz: %s",
                native_rate,
                e,
            )

        # Calculate chunk sizes
        stream_chunk_size = int(stream_rate * 0.08)  # 80ms at stream rate

        try:
            # Set up audio stream at chosen stream rate
            stream_kwargs = {
                "channels": 1,
                "samplerate": stream_rate,
                "dtype": "float32",
                "blocksize": 0,  # let sounddevice choose a safe blocksize
            }
            if device_to_use is not None:
                stream_kwargs["device"] = device_to_use
                logger.info("Opening stream with device %d", device_to_use)
            else:
                logger.info("Opening stream with default device")

            with sd.InputStream(**stream_kwargs) as stream:
                logger.info(
                    "Wake word audio stream opened at %d Hz (target %d Hz)",
                    stream_rate,
                    target_rate,
                )

                while not self._stop_event.is_set():
                    # Wait if paused
                    if not self._paused.wait(timeout=0.1):
                        continue

                    # Check stop again after wait
                    if self._stop_event.is_set():
                        break

                    # Read audio chunk
                    try:
                        audio_data, overflowed = stream.read(stream_chunk_size)
                        if overflowed:
                            logger.debug("Audio buffer overflow")
                            continue
                    except Exception as e:
                        logger.warning("Error reading audio: %s", e)
                        time.sleep(0.1)
                        continue

                    # Convert to format expected by OpenWakeWord
                    audio_array = audio_data.flatten()
                    if audio_array.size == 0:
                        continue
                    if np.issubdtype(audio_array.dtype, np.floating):
                        audio_float = audio_array.astype(np.float32)
                    else:
                        audio_float = audio_array.astype(np.float32) / 32768.0

                    # Debug: check if we're getting actual audio (before resampling)
                    audio_level = float(np.sqrt(np.mean(np.square(audio_float))) * 32767.0)

                    # Resample from native rate to 16kHz if needed
                    if stream_rate != target_rate:
                        # Calculate number of samples after resampling
                        num_samples = int(len(audio_float) * target_rate / stream_rate)
                        if num_samples <= 0:
                            continue
                        audio_float = signal.resample(audio_float, num_samples)

                    # Convert back to int16 for OpenWakeWord
                    audio_array = np.clip(audio_float * 32767.0, -32768, 32767).astype(np.int16)

                    # Run prediction
                    try:
                        prediction = self._model.predict(audio_array)

                        # Check if wake word detected
                        for model_name, score in prediction.items():
                            # Track max score for debug
                            if score > self._max_score_seen:
                                self._max_score_seen = score

                            # Send score update callback (with audio level for debugging)
                            if self.on_score_update:
                                try:
                                    self.on_score_update(model_name, score, audio_level, self.device_name)
                                except Exception:
                                    pass

                            # Debug log every 5 seconds with max score seen
                            now = time.time()
                            if now - self._last_debug_log >= 5.0:
                                logger.info(
                                    "Wake word '%s' - current: %.3f, max seen: %.3f, threshold: %.2f",
                                    model_name, score, self._max_score_seen, self.threshold
                                )
                                self._last_debug_log = now
                                self._max_score_seen = 0.0  # Reset max after logging

                            if score >= self.threshold:
                                # Check cooldown
                                if now - self._last_detection_time >= self._cooldown_seconds:
                                    self._last_detection_time = now
                                    logger.info(
                                        "Wake word '%s' TRIGGERED with confidence %.2f",
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
