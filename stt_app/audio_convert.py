from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .config import get_app_dir
from .ffmpeg_manager import ensure_ffmpeg, get_ffmpeg_path

logger = logging.getLogger(__name__)


class AudioConversionError(RuntimeError):
    """Raised when an audio conversion step fails."""


def convert_opus_to_mp3(source: Path) -> Path:
    """Convert an OPUS file to MP3 using FFmpeg.

    The converted file is placed in the application's temp directory.
    Returns the path to the MP3 file.
    """
    if not source.exists():
        raise AudioConversionError(f"Quelldatei nicht gefunden: {source}")

    # Stelle sicher, dass FFmpeg verfügbar ist (installiert es bei Bedarf)
    if not ensure_ffmpeg():
        raise AudioConversionError(
            "FFmpeg konnte nicht automatisch installiert werden. "
            "Bitte FFmpeg manuell installieren."
        )
    
    ffmpeg_binary = get_ffmpeg_path()

    temp_dir = get_app_dir() / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(prefix=f"{source.stem}-", suffix=".mp3", dir=temp_dir)
    os.close(fd)
    output_path = Path(temp_path)

    command = [
        ffmpeg_binary,
        "-y",
        "-i",
        str(source),
        "-vn",
        "-acodec",
        "libmp3lame",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    ]

    logger.info("Konvertiere OPUS zu MP3: %s -> %s", source, output_path)

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stderr:
            logger.debug("FFmpeg stderr: %s", result.stderr.strip())
    except subprocess.CalledProcessError as exc:
        output_path.unlink(missing_ok=True)
        stderr = exc.stderr.strip() if exc.stderr else ""
        logger.error("FFmpeg-Konvertierung fehlgeschlagen: %s", stderr)
        raise AudioConversionError("Konvertierung nach MP3 fehlgeschlagen.") from exc

    return output_path
