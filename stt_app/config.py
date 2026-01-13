import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import keyring

APP_NAME = "STTDesktop"
SERVICE_NAME = "STTDesktop"
API_KEY_USERNAME = "api_key"


def get_app_dir() -> Path:
    """Return an OS-appropriate application data directory.

    On Windows this uses %LOCALAPPDATA%/STTDesktop
    """
    base = os.getenv("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    path = Path(base) / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_config_path() -> Path:
    return get_app_dir() / "config.json"


@dataclass
class AppSettings:
    """Runtime and persisted settings for the app.

    Keep names stable for backward compatibility when persisted as JSON.
    """

    # Hotkeys
    toggle_hotkey: str = "alt+t"
    cancel_key: str = "esc"

    # Audio
    sample_rate_hz: int = 16000
    channels: int = 1
    bit_depth: int = 16  # informational; we use 16-bit PCM WAV
    input_device_index: Optional[int] = None

    # Recording behavior
    max_duration_seconds: int = 0
    silence_threshold_rms: float = 0.02  # 0..1 normalized RMS threshold
    silence_min_seconds: float = 1.5
    # Silence handling toggle: when False, recording will NOT auto-stop on silence
    stop_on_silence: bool = False

    # UX behavior
    auto_copy: bool = True
    auto_paste: bool = False
    auto_close_popup_seconds: int = 10
    history_limit: int = 20

    # Transcription
    model: str = "whisper-large-v3"
    response_format: str = "verbose_json"

    # Language (None/"auto" lets model detect)
    language: Optional[str] = None
    
    # Visualization settings
    particle_count: int = 600
    glow_intensity: float = 1.0
    particle_color_hue: int = 200  # Blue
    
    # Grammar correction
    auto_grammar_correction: bool = False  # Automatic grammar correction after transcription

    # Wake word settings
    wake_word_enabled: bool = False  # Enable always-listening wake word detection
    wake_word_model: str = "hey_jarvis"  # Which wake word to listen for
    wake_word_threshold: float = 0.5  # Detection confidence threshold (0-1)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(data: str) -> "AppSettings":
        raw = json.loads(data)
        return AppSettings(**raw)


def load_settings() -> AppSettings:
    cfg_path = get_config_path()
    if cfg_path.exists():
        try:
            loaded = AppSettings.from_json(cfg_path.read_text(encoding="utf-8"))
            # Migration: previous default was 60 seconds; interpret as unlimited now
            # Only change when user hasn't explicitly set a different value
            if getattr(loaded, "max_duration_seconds", 60) == 60:
                loaded.max_duration_seconds = 0
                save_settings(loaded)
            return loaded
        except Exception:
            # Fallback to defaults if config is corrupted
            pass
    s = AppSettings()
    save_settings(s)
    return s


def save_settings(settings: AppSettings) -> None:
    cfg_path = get_config_path()
    cfg_path.write_text(settings.to_json(), encoding="utf-8")


def set_api_key_secure(api_key: str) -> None:
    """Persist the Groq API key securely using the OS keyring.

    The `groq` SDK reads GROQ_API_KEY from environment. We also export to `os.environ`
    for child libraries that read the env var at runtime.
    """
    keyring.set_password(SERVICE_NAME, API_KEY_USERNAME, api_key)
    os.environ["GROQ_API_KEY"] = api_key


def get_api_key_secure() -> Optional[str]:
    api_key = keyring.get_password(SERVICE_NAME, API_KEY_USERNAME)
    if not api_key:
        # Fallback to environment variable if set
        api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    return api_key
