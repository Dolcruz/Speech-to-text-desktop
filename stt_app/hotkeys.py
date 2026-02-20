import threading
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class HotkeyManager:
    """Register and manage global hotkeys.

    Uses the `keyboard` package to listen system-wide.
    """

    def __init__(self) -> None:
        self._toggle_id: Optional[int] = None
        self._cancel_id: Optional[int] = None
        self._toggle_str: Optional[str] = None
        self._cancel_str: Optional[str] = None
        self._lock = threading.Lock()
        self._keyboard = None

    def _ensure_keyboard(self):
        if self._keyboard is None:
            import keyboard

            self._keyboard = keyboard
        return self._keyboard

    def _unregister_locked(self) -> None:
        keyboard = self._keyboard
        if self._toggle_id is not None:
            try:
                if keyboard is not None:
                    keyboard.remove_hotkey(self._toggle_id)
            except Exception:
                pass
            self._toggle_id = None
        if self._cancel_id is not None:
            try:
                if keyboard is not None:
                    keyboard.remove_hotkey(self._cancel_id)
            except Exception:
                pass
            self._cancel_id = None
        self._toggle_str = None
        self._cancel_str = None

    def register(self, toggle_hotkey: str, cancel_key: str, on_toggle: Callable[[], None], on_cancel: Callable[[], None]) -> bool:
        """Register global hotkeys. Returns True if registration succeeded.

        On Windows, global hotkeys may require Administrator privileges.
        If registration fails, the app will still work with keyboard shortcuts
        when the window has focus.
        """
        ok = True
        with self._lock:
            self._unregister_locked()
            self._toggle_str = toggle_hotkey
            self._cancel_str = cancel_key
            keyboard = self._ensure_keyboard()
            
            # Try to register toggle hotkey
            try:
                self._toggle_id = keyboard.add_hotkey(
                    toggle_hotkey, 
                    on_toggle, 
                    suppress=False, 
                    trigger_on_release=False
                )
                logger.info("Successfully registered toggle hotkey: %s", toggle_hotkey)
            except Exception as exc:
                logger.warning("Failed to register toggle hotkey '%s': %s", toggle_hotkey, exc)
                self._toggle_id = None
                ok = False
            
            # Try to register cancel key
            try:
                self._cancel_id = keyboard.add_hotkey(
                    cancel_key, 
                    on_cancel, 
                    suppress=False, 
                    trigger_on_release=False
                )
                logger.info("Successfully registered cancel key: %s", cancel_key)
            except Exception as exc:
                logger.warning("Failed to register cancel key '%s': %s", cancel_key, exc)
                self._cancel_id = None
                # Cancel key failure is not critical
                
        return ok and self._toggle_id is not None

    def unregister(self) -> None:
        with self._lock:
            self._unregister_locked()

    def send_paste(self) -> None:
        """Send Ctrl+V to paste clipboard into the focused window."""
        try:
            keyboard = self._ensure_keyboard()
            keyboard.send("ctrl+v")
        except Exception as exc:
            logger.warning("Failed to send paste: %s", exc)
