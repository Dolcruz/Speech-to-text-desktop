from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import List, Optional

from PySide6 import QtCore, QtWidgets, QtGui

from stt_app.config import load_settings, save_settings, get_api_key_secure
from stt_app.logger import configure_logging
from stt_app.audio import AudioRecorder, RecorderCallbacks
from stt_app.wakeword import WakeWordDetector
from stt_app.audio_convert import convert_opus_to_mp3, AudioConversionError
from stt_app.ffmpeg_manager import ensure_ffmpeg
from stt_app.groq_client import GroqTranscriber
from stt_app.hotkeys import HotkeyManager
from stt_app.ui_main import MainWindow
from stt_app.ui_overlay import RecordingOverlay
from stt_app.ui_result_popup import ResultPopup
from stt_app.ui_dialog import DialogWindow
from stt_app.ui_update import UpdateDialog
from stt_app.tts_client import TTSClient, VOICE_OPTIONS
from stt_app.updater import check_for_updates, download_update, install_update
from stt_app.theme import apply_dark_theme

logger = logging.getLogger(__name__)


class Controller(QtCore.QObject):
    """Glue layer connecting hotkeys, recorder, transcriber, and UI."""

    # Signal for wake word score updates (thread-safe) - model, score, audio_level, device_name
    wake_word_score_signal = QtCore.Signal(str, float, float, str)

    def __init__(self, app: QtWidgets.QApplication) -> None:
        super().__init__()
        self.app = app
        self.settings = load_settings()

        self.window = MainWindow(self.settings)
        self.overlay = RecordingOverlay()
        self.dialog_window: Optional[DialogWindow] = None
        
        # Apply saved visualization settings
        self.overlay._particle_sphere.set_particle_count(self.settings.particle_count)
        self.overlay._particle_sphere.set_glow_intensity(self.settings.glow_intensity)
        self.overlay._particle_sphere.set_color_hue(self.settings.particle_color_hue)
        
        self.hotkeys = HotkeyManager()
        self.transcriber = GroqTranscriber(self.settings)
        self.tts_client = TTSClient()  # No API key needed - edge-tts is free!
        
        # Dialog mode state
        self._dialog_mode_active = False
        self._dialog_current_lang = "Deutsch"

        # Wake word detector
        self._wake_word_detector: Optional[WakeWordDetector] = None
        self._last_score_update = 0.0
        self.wake_word_score_signal.connect(self._update_wake_word_status)
        self._setup_wake_word()

        # FFmpeg setup
        self._setup_ffmpeg()

        self.recorder = AudioRecorder(
            settings=self.settings,
            callbacks=RecorderCallbacks(
                on_level=self._on_level,
                on_time=self._on_time,
                on_stopped=self._on_stopped,
                on_cancelled=self._on_cancelled,
                on_error=self._on_record_error,
            ),
        )

        # Hook UI signals first
        self.window.start_stop_requested.connect(self.toggle_recording)
        self.window.cancel_requested.connect(self.cancel_recording)
        self.window.wake_word_settings_changed.connect(self._on_wake_word_settings_changed)
        self.window.correct_text_requested.connect(self.correct_history_text)
        self.window.dialog_mode_requested.connect(self.open_dialog_mode)
        self.window.file_transcription_requested.connect(self._transcribe_selected_audio)
        self.overlay.cancel_requested.connect(self.cancel_recording)
        
        # Hook visual settings button
        self.window._visual_settings_btn.clicked.disconnect()  # Remove default connection
        self.window._visual_settings_btn.clicked.connect(self.open_visual_settings)

        # Show window immediately so the app is visible, regardless of hotkey registration
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()
        QtCore.QTimer.singleShot(300, self.window.show_tray_tip)
        self.window.set_status("Bereit")
        
        # Check for updates after a short delay (non-blocking)
        QtCore.QTimer.singleShot(2000, self._check_for_updates)

        # Register global hotkeys in a background thread to avoid any UI freeze
        threading.Thread(target=self._register_hotkeys_bg, name="HotkeyRegister", daemon=True).start()
    
    def _setup_ffmpeg(self) -> None:
        """Stellt sicher, dass FFmpeg verfügbar ist"""
        def check_ffmpeg():
            try:
                logger.info("Prüfe FFmpeg-Verfügbarkeit...")
                if ensure_ffmpeg():
                    logger.info("FFmpeg ist verfügbar")
                    # Optional: UI-Benachrichtigung anzeigen
                    QtCore.QTimer.singleShot(0, lambda: self.window.set_status("FFmpeg bereit"))
                else:
                    logger.warning("FFmpeg konnte nicht installiert werden")
                    QtCore.QTimer.singleShot(0, lambda: self.window.set_status("FFmpeg-Installation fehlgeschlagen"))
            except Exception as e:
                logger.error("FFmpeg-Setup fehlgeschlagen: %s", e)
                QtCore.QTimer.singleShot(0, lambda: self.window.set_status("FFmpeg-Setup fehlgeschlagen"))

        # FFmpeg-Check im Hintergrund durchführen
        threading.Thread(target=check_ffmpeg, daemon=True).start()

    def _setup_wake_word(self) -> None:
        """Initialize wake word detector if enabled in settings."""
        if not self.settings.wake_word_enabled:
            logger.info("Wake word detection disabled")
            self.window.hide_wake_word_score()
            return

        # Get device name for initial display
        import sounddevice as sd
        try:
            if self.settings.input_device_index is not None:
                device_info = sd.query_devices(self.settings.input_device_index)
                init_device_name = device_info.get('name', f'Device {self.settings.input_device_index}')
            else:
                device_info = sd.query_devices(kind='input')
                init_device_name = device_info.get('name', 'Default')
        except Exception:
            init_device_name = "Unbekannt"

        # Show initial "loading" state immediately
        self.window.set_wake_word_score(self.settings.wake_word_model, 0.0, self.settings.wake_word_threshold, 0.0, init_device_name)

        def setup():
            try:
                logger.info("Setting up wake word detector...")
                # Force 16kHz for OpenWakeWord
                self._wake_word_detector = WakeWordDetector(
                    wake_word=self.settings.wake_word_model,
                    threshold=self.settings.wake_word_threshold,
                    on_detected=self._on_wake_word_detected,
                    on_score_update=self._on_wake_word_score,
                    sample_rate=16000,  # OpenWakeWord requires exactly 16kHz
                    input_device_index=self.settings.input_device_index,
                )
                if self._wake_word_detector.start():
                    logger.info("Wake word detector started successfully")
                else:
                    error = self._wake_word_detector.get_model_error() or "Unbekannter Fehler"
                    logger.error("Wake word detector failed to start: %s", error)
            except Exception as e:
                logger.error("Wake word setup failed: %s", e)

        # Setup in background to avoid blocking UI
        threading.Thread(target=setup, daemon=True, name="WakeWordSetup").start()

    def _on_wake_word_score(self, model_name: str, score: float, audio_level: float, device_name: str) -> None:
        """Called with live score updates from wake word detector (from background thread)."""
        # Only update UI every 200ms to avoid flooding
        now = time.time()
        if now - self._last_score_update >= 0.2:
            self._last_score_update = now
            # Emit signal to update UI on main thread
            self.wake_word_score_signal.emit(model_name, score, audio_level, device_name)

    @QtCore.Slot(str, float, float, str)
    def _update_wake_word_status(self, model_name: str, score: float, audio_level: float, device_name: str) -> None:
        """Update wake word score display (runs on main thread)."""
        self.window.set_wake_word_score(model_name, score, self.settings.wake_word_threshold, audio_level, device_name)

    def _on_wake_word_detected(self) -> None:
        """Called when wake word is detected - start recording."""
        logger.info("Wake word detected! Starting recording...")
        # Invoke toggle_recording on UI thread
        QtCore.QMetaObject.invokeMethod(
            self, "toggle_recording", QtCore.Qt.QueuedConnection
        )

    @QtCore.Slot()
    def _on_wake_word_settings_changed(self) -> None:
        """Handle wake word settings changes from UI."""
        logger.info("Wake word settings changed, reconfiguring...")

        # Stop existing detector
        if self._wake_word_detector is not None:
            self._wake_word_detector.stop()
            self._wake_word_detector = None

        # Hide label first
        self.window.hide_wake_word_score()

        # Setup new detector if enabled
        self._setup_wake_word()

    def _register_hotkeys_bg(self) -> None:
        def do_register():
            # Ensure hotkey handlers run on the UI thread even if the hotkey
            # library invokes them from a background thread.
            def invoke_toggle() -> None:
                QtCore.QMetaObject.invokeMethod(self, "toggle_recording", QtCore.Qt.QueuedConnection)

            def invoke_cancel() -> None:
                QtCore.QMetaObject.invokeMethod(self, "cancel_recording", QtCore.Qt.QueuedConnection)

            ok = self.hotkeys.register(
                toggle_hotkey=self.settings.toggle_hotkey,
                cancel_key=self.settings.cancel_key,
                on_toggle=invoke_toggle,
                on_cancel=invoke_cancel,
            )
            if not ok:
                # Inform user that global hotkeys are not available
                # but local shortcuts still work when window is focused
                QtCore.QMetaObject.invokeMethod(
                    self.window,
                    "set_status",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, "Bereit • Für systemweite Hotkeys App als Administrator starten"),
                )
            else:
                # Global hotkeys are active
                QtCore.QMetaObject.invokeMethod(
                    self.window,
                    "set_status",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, "Bereit • Globale Hotkeys aktiv"),
                )
        # Delay a bit so UI can settle
        time.sleep(0.4)
        do_register()

    # Event handlers from recorder (called on recorder thread)
    def _on_level(self, level: float) -> None:
        QtCore.QMetaObject.invokeMethod(
            self.overlay, "update_level", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(float, level)
        )

    def _on_time(self, seconds: float) -> None:
        QtCore.QMetaObject.invokeMethod(
            self.overlay, "update_time", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(float, seconds)
        )

    def _on_stopped(self, wav_path: Path) -> None:
        # Switch to processing state on UI
        QtCore.QMetaObject.invokeMethod(
            self, "_start_transcription", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, str(wav_path))
        )

    def _on_cancelled(self) -> None:
        QtCore.QMetaObject.invokeMethod(self, "_cancelled_ui", QtCore.Qt.QueuedConnection)

    @QtCore.Slot()
    def _cancelled_ui(self) -> None:
        self.overlay.hide()
        self.window.set_status("Abgebrochen")
        self.window.set_recording_state(False)
        # Resume wake word detection
        if self._wake_word_detector is not None:
            self._wake_word_detector.resume()

    def _on_record_error(self, message: str) -> None:
        QtCore.QMetaObject.invokeMethod(
            self, "_record_error_ui", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, message)
        )

    @QtCore.Slot(str)
    def _record_error_ui(self, message: str) -> None:
        self.overlay.hide()
        self.window.set_status(f"Audiofehler: {message}")
        self.window.set_recording_state(False)
        # Resume wake word detection after error
        if self._wake_word_detector is not None:
            self._wake_word_detector.resume()

    def _transcribe_audio_path(self, audio_path: Path, cleanup_after: bool) -> None:
        audio_path = audio_path.expanduser()
        if audio_path.exists():
            try:
                audio_path = audio_path.resolve()
            except Exception:
                pass
        else:
            self.window.set_status(f"Datei nicht gefunden: {audio_path}")
            return

        display_name = audio_path.name
        remove_after = cleanup_after

        if audio_path.suffix.lower() == ".opus":
            self.window.set_status(f"Konvertiere {display_name}...")
            try:
                audio_path = convert_opus_to_mp3(audio_path)
                remove_after = True
                display_name = f"{display_name} (konvertiert)"
            except AudioConversionError as exc:
                self.overlay.hide()
                self.window.set_recording_state(False)
                self.window.set_status(f"Konvertierung fehlgeschlagen: {exc}")
                return

        self.overlay.hide()
        self.window.set_status(f"Transkribiere {display_name}...")
        self.window.set_recording_state(False)

        def worker(path: Path, remove_after: bool) -> None:
            text = ""
            try:
                result = self.transcriber.transcribe_wav(path)
                text = result.text or ""
            except Exception as exc:
                text = f"Fehler bei der Transkription: {exc}"
            finally:
                if remove_after:
                    try:
                        path.unlink(missing_ok=True)
                    except Exception:
                        pass
            QtCore.QMetaObject.invokeMethod(
                self, "_show_result", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, text)
            )

        threading.Thread(
            target=worker,
            args=(audio_path, remove_after),
            name="TranscriptionThread",
            daemon=True,
        ).start()

    @QtCore.Slot(str)
    def _start_transcription(self, wav_path_str: str) -> None:
        self._transcribe_audio_path(Path(wav_path_str), cleanup_after=True)

    @QtCore.Slot(list)
    def _transcribe_selected_audio(self, file_paths: List[str]) -> None:
        """Handle multiple files selected for transcription."""
        if not file_paths:
            return
            
        # Convert all to Path objects
        paths = [Path(p) for p in file_paths]
        
        # Start batch processing
        self._process_batch(paths)

    def _process_batch(self, file_paths: List[Path]) -> None:
        """Process a batch of files sequentially in a background thread."""
        self.overlay.hide()
        self.window.set_recording_state(False)
        
        def worker(paths: List[Path]) -> None:
            results = []
            total = len(paths)
            
            for i, path in enumerate(paths, 1):
                # Update status
                msg = f"Verarbeite Datei {i} von {total}: {path.name}..."
                QtCore.QMetaObject.invokeMethod(
                    self.window, "set_status", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, msg)
                )
                
                # Process single file (synchronous logic extracted here)
                try:
                    text = self._transcribe_single_file_sync(path)
                    if text:
                        results.append(f"--- {path.name} ---\n{text}")
                except Exception as e:
                    results.append(f"--- {path.name} ---\nFehler: {e}")
            
            # Combine results
            final_text = "\n\n".join(results)
            
            # Show result
            QtCore.QMetaObject.invokeMethod(
                self, "_show_result", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, final_text)
            )

        threading.Thread(
            target=worker,
            args=(file_paths,),
            name="BatchTranscriptionThread",
            daemon=True,
        ).start()

    def _transcribe_single_file_sync(self, audio_path: Path) -> str:
        """Synchronous helper to transcribe a single file (runs in worker thread)."""
        audio_path = audio_path.expanduser()
        if audio_path.exists():
            try:
                audio_path = audio_path.resolve()
            except Exception:
                pass
        else:
            return f"Datei nicht gefunden: {audio_path}"

        display_name = audio_path.name
        temp_file_created = False

        try:
            # Convert opus if needed
            if audio_path.suffix.lower() == ".opus":
                QtCore.QMetaObject.invokeMethod(
                    self.window, "set_status", QtCore.Qt.QueuedConnection, 
                    QtCore.Q_ARG(str, f"Konvertiere {display_name}...")
                )
                audio_path = convert_opus_to_mp3(audio_path)
                temp_file_created = True
                display_name = f"{display_name} (konvertiert)"

            # Transcribe
            QtCore.QMetaObject.invokeMethod(
                self.window, "set_status", QtCore.Qt.QueuedConnection, 
                QtCore.Q_ARG(str, f"Transkribiere {display_name}...")
            )
            
            result = self.transcriber.transcribe_wav(audio_path)
            return result.text or ""
            
        except Exception as exc:
            return f"Fehler: {exc}"
        finally:
            if temp_file_created:
                try:
                    audio_path.unlink(missing_ok=True)
                except Exception:
                    pass


    @QtCore.Slot(str)
    def _show_result(self, text: str) -> None:
        self.window.set_status("Fertig")
        self.window.set_recording_state(False)
        
        # If in dialog mode, handle differently
        if self._dialog_mode_active and self.dialog_window:
            self._handle_dialog_result(text)
            return
        
        # Normal mode: check if automatic grammar correction is enabled
        if self.settings.auto_grammar_correction:
            # Automatically correct grammar before showing popup
            self.window.set_status("Korrigiere Grammatik automatisch...")
            corrected_text = self._correct_grammar_sync(text)
            final_text = corrected_text
            show_correct_btn = False  # Don't show button since we already corrected
        else:
            final_text = text
            show_correct_btn = True  # Show button for manual correction
        
        popup = ResultPopup(
            text=final_text, 
            auto_close_seconds=self.settings.auto_close_popup_seconds, 
            show_correct_button=show_correct_btn,
            parent=self.window
        )
        
        # Connect process signal (correct & optionally translate)
        popup.process_requested.connect(lambda text, lang: self._on_process_text(popup, text, lang))
        
        if self.settings.auto_copy:
            QtWidgets.QApplication.clipboard().setText(final_text)
        if self.settings.auto_paste:
            # Paste after brief delay to allow popup to render (avoid pasting into popup)
            QtCore.QTimer.singleShot(200, lambda: self.hotkeys.send_paste())
        
        self.window.set_status("Fertig")
        popup.exec()
        # Append to history
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.window.append_history(timestamp, popup.get_text(), self.settings.history_limit)

        # Resume wake word detection after popup is closed
        if self._wake_word_detector is not None:
            self._wake_word_detector.resume()

    def _correct_grammar_sync(self, text: str) -> str:
        """Synchronously correct grammar of text (runs in UI thread)."""
        try:
            return self.transcriber.correct_grammar(text)
        except Exception as e:
            logging.getLogger(__name__).error(f"Grammar correction failed: {e}")
            return text  # Return original text on error
    
    def _on_process_text(self, popup: ResultPopup, text: str, target_language: str) -> None:
        """Handle combined correction and optional translation request.
        
        If target_language is empty, only corrects grammar.
        If target_language is specified, corrects grammar first, then translates.
        """
        def worker():
            try:
                # Step 1: Always correct grammar first
                corrected = self.transcriber.correct_grammar(text)
                
                # Step 2: If target language specified, translate the corrected text
                if target_language:
                    final_text = self.transcriber.translate_text(corrected, target_language)
                else:
                    final_text = corrected
                
                # Update popup with final processed text on UI thread
                QtCore.QMetaObject.invokeMethod(
                    popup, "set_processed_text", QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, final_text),
                    QtCore.Q_ARG(str, target_language)
                )
            except Exception as e:
                logging.getLogger(__name__).error(f"Text processing failed: {e}")
                QtCore.QMetaObject.invokeMethod(
                    popup._status_label, "setText", QtCore.Qt.QueuedConnection, 
                    QtCore.Q_ARG(str, f"Fehler: {e}")
                )
                QtCore.QMetaObject.invokeMethod(
                    popup._process_btn, "setEnabled", QtCore.Qt.QueuedConnection, 
                    QtCore.Q_ARG(bool, True)
                )
        
        threading.Thread(target=worker, name="ProcessTextThread", daemon=True).start()

    @QtCore.Slot(str, object)
    def correct_history_text(self, text: str, dialog: object) -> None:
        """Handle combined correction & translation request from history dialog."""
        def worker():
            try:
                # Get target language from dialog (empty string = no translation)
                target_language = getattr(dialog, '_target_language', '')
                
                # Step 1: Always correct grammar first
                corrected = self.transcriber.correct_grammar(text)
                
                # Step 2: If target language specified, translate the corrected text
                if target_language:
                    final_text = self.transcriber.translate_text(corrected, target_language)
                else:
                    final_text = corrected
                
                # Call success callback on UI thread using QTimer
                if hasattr(dialog, '_process_callback'):
                    QtCore.QTimer.singleShot(0, lambda: dialog._process_callback(final_text, target_language))
            except Exception as e:
                logging.getLogger(__name__).error(f"History text processing failed: {e}")
                # Call error callback on UI thread using QTimer
                if hasattr(dialog, '_process_error_callback'):
                    QtCore.QTimer.singleShot(0, lambda: dialog._process_error_callback(str(e)))
        
        threading.Thread(target=worker, name="HistoryProcessTextThread", daemon=True).start()
    
    # Dialog mode methods
    @QtCore.Slot()
    def open_dialog_mode(self) -> None:
        """Open the dialog mode window for two-way translation."""
        if self.dialog_window is not None:
            # Already open, just bring to front
            self.dialog_window.show()
            self.dialog_window.raise_()
            self.dialog_window.activateWindow()
            return
        
        # Create dialog window (no API key needed - TTS is free!)
        self.dialog_window = DialogWindow(self.window)
        self.dialog_window.start_recording.connect(self._dialog_start_recording)
        self.dialog_window.stop_recording.connect(self._dialog_stop_recording)
        self.dialog_window.closed.connect(self._dialog_closed)
        self.dialog_window.show()
    
    @QtCore.Slot(str)
    def _dialog_start_recording(self, current_lang: str) -> None:
        """Start recording for dialog mode."""
        self._dialog_mode_active = True
        self._dialog_current_lang = current_lang
        self.dialog_window.set_status(f"Aufnahme läuft ({current_lang})...")
        
        # Use standard recording with overlay
        if not self.recorder.is_recording():
            self.recorder.start()
            self.overlay.show()  # Show overlay, not start()
    
    @QtCore.Slot()
    def _dialog_stop_recording(self) -> None:
        """Stop recording for dialog mode."""
        if self.recorder.is_recording():
            self.recorder.stop()
    
    @QtCore.Slot()
    def _dialog_closed(self) -> None:
        """Handle dialog window close."""
        self._dialog_mode_active = False
        self.dialog_window = None
    
    def _handle_dialog_result(self, original_text: str) -> None:
        """Handle transcription result in dialog mode."""
        if not self.dialog_window:
            return
        
        self.dialog_window.set_status("Übersetze...")
        
        def worker():
            try:
                # Get target language from dialog window
                target_lang = self.dialog_window.get_target_language()
                current_speaker = self.dialog_window.get_current_speaker()
                
                # Step 1: Correct grammar
                corrected = self.transcriber.correct_grammar(original_text)
                
                # Step 2: Translate to target language
                translated = self.transcriber.translate_text(corrected, target_lang)
                
                # Step 3: Update UI on main thread
                QtCore.QMetaObject.invokeMethod(
                    self.dialog_window, "set_status", QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, f"Spreche aus ({target_lang})...")
                )
                
                # Step 4: Text-to-Speech and play (blocking)
                voice = VOICE_OPTIONS.get(target_lang, "de-DE-KatjaNeural")
                self.tts_client.text_to_speech_and_play(translated, voice=voice)
                
                # Step 5: Add to history and switch speaker
                QtCore.QMetaObject.invokeMethod(
                    self.dialog_window, "add_to_history", QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, f"Sprecher {current_speaker}"),
                    QtCore.Q_ARG(str, self._dialog_current_lang),
                    QtCore.Q_ARG(str, original_text),
                    QtCore.Q_ARG(str, translated)
                )
                
                QtCore.QMetaObject.invokeMethod(
                    self.dialog_window, "auto_switch_speaker", QtCore.Qt.QueuedConnection
                )
                
                QtCore.QMetaObject.invokeMethod(
                    self.dialog_window, "reset_recording_state", QtCore.Qt.QueuedConnection
                )
                
                QtCore.QMetaObject.invokeMethod(
                    self.dialog_window, "set_status", QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, "Bereit für nächste Aufnahme")
                )
                
            except Exception as e:
                logging.getLogger(__name__).error(f"Dialog processing failed: {e}")
                QtCore.QMetaObject.invokeMethod(
                    self.dialog_window, "set_status", QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, f"Fehler: {e}")
                )
                QtCore.QMetaObject.invokeMethod(
                    self.dialog_window, "reset_recording_state", QtCore.Qt.QueuedConnection
                )
        
        threading.Thread(target=worker, name="DialogProcessThread", daemon=True).start()

    # Public controls
    @QtCore.Slot()
    def toggle_recording(self) -> None:
        if self.recorder.is_recording():
            self.window.set_status("Stoppe Aufnahme…")
            self.window.set_recording_state(False)
            self.recorder.stop()
            return
        # Pause wake word detection while recording to avoid conflicts
        if self._wake_word_detector is not None:
            self._wake_word_detector.pause()
        # Start
        started = self.recorder.start()
        if started:
            self.overlay.show_top_right()
            self.window.set_status("Aufnahme läuft… (Alt+T zum Stoppen)")
            self.window.set_recording_state(True)

    @QtCore.Slot()
    def cancel_recording(self) -> None:
        if self.recorder.is_recording():
            self.window.set_status("Abbreche…")
            self.window.set_recording_state(False)
            self.recorder.cancel()
            # Resume wake word detection after cancel
            if self._wake_word_detector is not None:
                self._wake_word_detector.resume()
    
    @QtCore.Slot()
    def open_visual_settings(self) -> None:
        """Open visual settings dialog with REAL microphone preview."""
        from stt_app.ui_visual_settings import VisualSettingsDialog
        
        # Show overlay for live preview
        self.overlay.show_top_right()
        
        # Start REAL audio monitoring for preview (like actual recording but don't save)
        # Use the recorder's audio monitoring without actually recording
        import sounddevice as sd
        import numpy as np
        
        self._preview_stream = None
        
        def audio_callback(indata, frames, time_info, status):
            """Process real microphone input for preview."""
            if status:
                return
            # Calculate RMS level from audio data (same as real recording)
            audio_data = indata[:, 0] if len(indata.shape) > 1 else indata
            rms = float(np.sqrt(np.mean(audio_data**2)))
            # Normalize to 0-1 range (same normalization as recorder)
            normalized_level = min(1.0, rms * 50)  # Scale factor like in audio.py
            # Update overlay with REAL audio level
            QtCore.QMetaObject.invokeMethod(
                self.overlay._particle_sphere, 
                "set_level", 
                QtCore.Qt.QueuedConnection, 
                QtCore.Q_ARG(float, normalized_level)
            )
        
        try:
            # Start audio stream with same settings as recorder
            self._preview_stream = sd.InputStream(
                samplerate=self.settings.sample_rate_hz,
                channels=self.settings.channels,
                callback=audio_callback,
                blocksize=int(self.settings.sample_rate_hz * 0.1),  # 100ms blocks
                device=self.settings.input_device_index
            )
            self._preview_stream.start()
        except Exception as e:
            # Fallback if audio fails
            import logging
            logging.getLogger(__name__).warning("Could not start audio preview: %s", e)
        
        # Create and show settings dialog with current settings
        dialog = VisualSettingsDialog(
            particle_count=self.settings.particle_count,
            glow_intensity=self.settings.glow_intensity,
            color_hue=self.settings.particle_color_hue,
            parent=self.window
        )
        
        # Connect live update signals
        dialog.particle_count_changed.connect(self.overlay._particle_sphere.set_particle_count)
        dialog.glow_intensity_changed.connect(self.overlay._particle_sphere.set_glow_intensity)
        dialog.color_hue_changed.connect(self.overlay._particle_sphere.set_color_hue)
        
        # Execute dialog
        result = dialog.exec()
        
        # Stop REAL audio preview
        if self._preview_stream:
            try:
                self._preview_stream.stop()
                self._preview_stream.close()
            except Exception:
                pass
            self._preview_stream = None
        
        self.overlay.hide()
        
        # Save settings if user clicked OK
        if result == QtWidgets.QDialog.Accepted:
            count, glow, hue = dialog.get_values()
            self.settings.particle_count = count
            self.settings.glow_intensity = glow
            self.settings.particle_color_hue = hue
            save_settings(self.settings)
            self.window.set_status("Visualisierungs-Einstellungen gespeichert")
        else:
            # Reset to saved settings if cancelled
            self.overlay._particle_sphere.set_particle_count(self.settings.particle_count)
            self.overlay._particle_sphere.set_glow_intensity(self.settings.glow_intensity)
            self.overlay._particle_sphere.set_color_hue(self.settings.particle_color_hue)
            self.window.set_status("Einstellungen verworfen")
    
    # Auto-Update System
    def _check_for_updates(self) -> None:
        """Check for updates in background thread."""
        import sys
        
        # Only check for updates if running as .exe
        if not getattr(sys, 'frozen', False):
            logging.getLogger(__name__).info("Running from source - skipping update check")
            return
        
        def worker():
            try:
                update_info = check_for_updates()
                if update_info:
                    new_version, download_url, release_notes = update_info
                    # Show update dialog on UI thread
                    QtCore.QMetaObject.invokeMethod(
                        self, "_show_update_dialog", QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(str, new_version),
                        QtCore.Q_ARG(str, download_url),
                        QtCore.Q_ARG(str, release_notes)
                    )
            except Exception as e:
                logging.getLogger(__name__).error(f"Error checking for updates: {e}")
        
        threading.Thread(target=worker, name="UpdateCheckThread", daemon=True).start()
    
    @QtCore.Slot(str, str, str)
    def _show_update_dialog(self, new_version: str, download_url: str, release_notes: str) -> None:
        """Show update dialog to user."""
        from stt_app.updater import get_current_version
        
        current_version = get_current_version()
        dialog = UpdateDialog(
            current_version=current_version,
            new_version=new_version,
            download_url=download_url,
            release_notes=release_notes,
            parent=self.window
        )
        
        # Connect download signal
        dialog.download_requested.connect(lambda url: self._download_update(dialog, url))
        
        dialog.exec()
    
    def _download_update(self, dialog: UpdateDialog, download_url: str) -> None:
        """Download update in background thread."""
        def progress_callback(downloaded: int, total: int):
            QtCore.QMetaObject.invokeMethod(
                dialog, "update_progress", QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(int, downloaded),
                QtCore.Q_ARG(int, total)
            )
        
        def worker():
            try:
                # Download update
                update_file = download_update(download_url, progress_callback)
                
                if update_file:
                    # Notify completion
                    QtCore.QMetaObject.invokeMethod(
                        dialog, "download_complete", QtCore.Qt.QueuedConnection
                    )
                    
                    # Wait a moment for user to see completion message
                    time.sleep(1)
                    
                    # Install update (this will close the app)
                    if install_update(update_file):
                        # Close dialog and exit app
                        QtCore.QMetaObject.invokeMethod(
                            dialog, "accept", QtCore.Qt.QueuedConnection
                        )
                        QtCore.QMetaObject.invokeMethod(
                            self.app, "quit", QtCore.Qt.QueuedConnection
                        )
                else:
                    QtCore.QMetaObject.invokeMethod(
                        dialog, "download_error", QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(str, "Download fehlgeschlagen")
                    )
            except Exception as e:
                logging.getLogger(__name__).error(f"Error downloading update: {e}")
                QtCore.QMetaObject.invokeMethod(
                    dialog, "download_error", QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, str(e))
                )
        
        threading.Thread(target=worker, name="UpdateDownloadThread", daemon=True).start()


def main() -> None:
    log_path = configure_logging()
    logging.getLogger(__name__).info("Starting app; logs at %s", log_path)

    # High DPI support - removed deprecated attributes
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts, True)

    app = QtWidgets.QApplication([])
    apply_dark_theme(app)
    controller = Controller(app)
    app.exec()


if __name__ == "__main__":
    main()
