from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import soundfile as sf

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from groq import Groq

from .config import get_api_key_secure, AppSettings

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 24 * 1024 * 1024  # 24 MB safety limit to stay under provider cap
WAV_HEADER_BYTES = 4096  # Approximate WAV header size to keep chunks within limit

# SECURITY NOTE: No fallback API key included for security reasons.
# Users must configure their own Groq API key via the UI settings or environment variable GROQ_API_KEY.
FALLBACK_API_KEY = None


@dataclass
class TranscriptionResult:
    text: str
    raw: dict


class GroqTranscriber:
    """Wrapper around Groq SDK for Whisper transcription.

    Uses retries on transient failures and returns the transcribed text.
    Lazily initializes the underlying client to avoid UI-thread stalls.
    """

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._client: Optional[Groq] = None

    def _ensure_client(self) -> Groq:
        if self._client is not None:
            return self._client
        api_key = get_api_key_secure()
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not configured. Please set your API key via:\n"
                "1. UI Settings (Main Window → Groq API Key field)\n"
                "2. Environment variable: GROQ_API_KEY=your_key_here\n"
                "Get your API key from: https://console.groq.com/keys"
            )
        logger.info("Using GROQ_API_KEY from keyring/env.")
        self._client = Groq(api_key=api_key)
        return self._client

    def _prepare_audio_for_upload(self, audio_path: Path) -> Tuple[List[Path], Optional[Path]]:
        size = audio_path.stat().st_size
        if size <= MAX_UPLOAD_BYTES:
            return [audio_path], None

        temp_dir = Path(tempfile.mkdtemp(prefix="groq-chunks-"))
        chunk_paths: List[Path] = []
        try:
            with sf.SoundFile(str(audio_path), "r") as src:
                channels = max(1, getattr(src, "channels", 1) or 1)
                samplerate = getattr(src, "samplerate", 0) or self.settings.sample_rate_hz or 16000
                bytes_per_frame = max(1, 2 * channels)
                max_frames = max(1, (MAX_UPLOAD_BYTES - WAV_HEADER_BYTES) // bytes_per_frame)

                index = 0
                while True:
                    data = src.read(frames=max_frames, dtype="float32")
                    if data.size == 0:
                        break

                    chunk_path = temp_dir / f"{audio_path.stem}_part{index:03d}.wav"
                    with sf.SoundFile(
                        str(chunk_path),
                        mode="w",
                        samplerate=samplerate,
                        channels=channels,
                        subtype="PCM_16",
                        format="WAV",
                    ) as dest:
                        dest.write(data)

                    chunk_paths.append(chunk_path)
                    index += 1
        except Exception as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise ValueError(
                f"Audio '{audio_path.name}' is too large and could not be chunked automatically: {exc}"
            ) from exc

        if not chunk_paths:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return [audio_path], None

        logger.info(
            "Split oversized audio '%s' (%s bytes) into %s PCM chunks for upload.",
            audio_path,
            size,
            len(chunk_paths),
        )
        return chunk_paths, temp_dir

    def _transcribe_file(
        self,
        client: Groq,
        file_path: Path,
        model: str,
        response_format: str,
        language: Optional[str],
    ) -> TranscriptionResult:
        with open(file_path, "rb") as fp:
            transcription = client.audio.transcriptions.create(
                file=fp,
                model=model,
                response_format=response_format,
                **({"language": language} if language else {}),
            )

        text = getattr(transcription, "text", "")

        try:
            raw = json.loads(transcription.json())  # type: ignore[attr-defined]
        except Exception:
            raw = {"text": text}

        return TranscriptionResult(text=text, raw=raw)

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
    )
    def transcribe_wav(self, audio_path: Path) -> TranscriptionResult:
        client = self._ensure_client()
        model = self.settings.model
        response_format = self.settings.response_format
        language = self.settings.language

        chunks, temp_dir = self._prepare_audio_for_upload(audio_path)
        cleanup_dir = temp_dir
        try:
            if len(chunks) == 1:
                return self._transcribe_file(client, chunks[0], model, response_format, language)

            segment_results: List[TranscriptionResult] = []
            for index, chunk_path in enumerate(chunks):
                logger.info(
                    "Transcribing chunk %s/%s: %s",
                    index + 1,
                    len(chunks),
                    chunk_path.name,
                )
                segment_results.append(
                    self._transcribe_file(client, chunk_path, model, response_format, language)
                )

            combined_parts = [segment.text.strip() for segment in segment_results if segment.text.strip()]
            combined_text = "\n\n".join(combined_parts).strip()
            if not combined_text:
                combined_text = " ".join(segment.text for segment in segment_results).strip()

            combined_raw = {
                "text": combined_text,
                "source_file": audio_path.name,
                "chunks": [
                    {
                        "index": idx,
                        "filename": chunks[idx].name,
                        "text": segment_results[idx].text,
                        "raw": segment_results[idx].raw,
                    }
                    for idx in range(len(segment_results))
                ],
            }

            logger.info(
                "Combined %s transcription chunks for '%s'.",
                len(segment_results),
                audio_path.name,
            )
            return TranscriptionResult(text=combined_text, raw=combined_raw)
        finally:
            if cleanup_dir:
                shutil.rmtree(cleanup_dir, ignore_errors=True)


    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
    )
    def correct_grammar(self, text: str) -> str:
        """Correct grammar of the provided text using Groq's LLM.
        
        Uses the kimi-k2-instruct model to correct grammar while preserving meaning.
        Returns the corrected text only, without any additional commentary.
        """
        client = self._ensure_client()
        
        # Craft a precise prompt that ensures ONLY corrected text is returned
        system_prompt = (
            "Du bist ein präziser Grammatik-Korrektor. "
            "Korrigiere NUR die Grammatik, Rechtschreibung und Zeichensetzung des Textes. "
            "Gib AUSSCHLIESSLICH den korrigierten Text zurück, ohne Kommentare, Erklärungen oder zusätzliche Formatierung. "
            "Verändere den Inhalt oder die Bedeutung NICHT."
        )
        
        user_message = f"Korrigiere diesen Text: {text}"
        
        # Collect the streamed response
        corrected_text = ""
        try:
            completion = client.chat.completions.create(
                model="moonshotai/kimi-k2-instruct-0905",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,  # Lower temperature for more consistent corrections
                max_completion_tokens=4096,
                top_p=1,
                stream=True,
                stop=None
            )
            
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    corrected_text += chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Grammar correction failed: {e}")
            # Return original text on error
            return text
            
        # Return the corrected text, stripped of any leading/trailing whitespace
        result = corrected_text.strip()
        logger.info(f"Grammar correction: '{text}' → '{result}'")
        return result if result else text

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
    )
    def translate_text(self, text: str, target_language: str) -> str:
        """Translate the provided text to the target language using Groq's LLM.
        
        Uses the kimi-k2-instruct model to translate text accurately.
        Returns ONLY the translated text without any additional commentary.
        
        Args:
            text: The text to translate
            target_language: Target language (e.g., "Englisch", "Spanisch", "Arabisch", etc.)
        
        Returns:
            The translated text
        """
        client = self._ensure_client()
        
        # Craft a precise prompt that ensures ONLY translated text is returned
        system_prompt = (
            "Du bist ein präziser Übersetzer. "
            f"Übersetze den folgenden Text AUSSCHLIESSLICH in {target_language}. "
            "Gib NUR die Übersetzung zurück, ohne Kommentare, Erklärungen oder zusätzliche Formatierung. "
            "Behalte die Bedeutung und den Ton des Originaltextes bei."
        )
        
        user_message = f"Übersetze diesen Text in {target_language}: {text}"
        
        # Collect the streamed response
        translated_text = ""
        try:
            completion = client.chat.completions.create(
                model="moonshotai/kimi-k2-instruct-0905",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,  # Lower temperature for more consistent translations
                max_completion_tokens=4096,
                top_p=1,
                stream=True,
                stop=None
            )
            
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    translated_text += chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            # Return original text on error
            return text
            
        # Return the translated text, stripped of any leading/trailing whitespace
        result = translated_text.strip()
        logger.info(f"Translation to {target_language}: '{text}' → '{result}'")
        return result if result else text
