"""
FFmpeg Manager - Automatische Installation und Verwaltung von FFmpeg
"""

import logging
import os
import platform
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

from .config import get_app_dir

logger = logging.getLogger(__name__)


class FFmpegManager:
    """Verwaltet die automatische Installation von FFmpeg"""
    
    def __init__(self):
        self.app_dir = get_app_dir()
        self.ffmpeg_dir = self.app_dir / "ffmpeg"
        self.ffmpeg_exe = None
        self._setup_ffmpeg_path()
    
    def _setup_ffmpeg_path(self):
        """Setzt den Pfad zur FFmpeg-Executable"""
        system = platform.system().lower()
        
        if system == "windows":
            self.ffmpeg_exe = self.ffmpeg_dir / "bin" / "ffmpeg.exe"
        elif system == "darwin":  # macOS
            self.ffmpeg_exe = self.ffmpeg_dir / "bin" / "ffmpeg"
        else:  # Linux
            self.ffmpeg_exe = self.ffmpeg_dir / "bin" / "ffmpeg"
    
    def is_ffmpeg_available(self) -> bool:
        """Prüft ob FFmpeg verfügbar ist (systemweit oder lokal installiert)"""
        # Erst prüfen ob systemweit verfügbar
        if shutil.which("ffmpeg"):
            logger.info("FFmpeg systemweit verfügbar")
            return True
        
        # Dann prüfen ob lokal installiert
        if self.ffmpeg_exe and self.ffmpeg_exe.exists():
            logger.info("FFmpeg lokal installiert: %s", self.ffmpeg_exe)
            return True
        
        return False
    
    def get_ffmpeg_path(self) -> str:
        """Gibt den Pfad zur FFmpeg-Executable zurück"""
        # Erst systemweit suchen
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg:
            return system_ffmpeg
        
        # Dann lokal
        if self.ffmpeg_exe and self.ffmpeg_exe.exists():
            return str(self.ffmpeg_exe)
        
        raise RuntimeError("FFmpeg nicht verfügbar")
    
    def _get_download_url(self) -> str:
        """Gibt die Download-URL für FFmpeg basierend auf dem Betriebssystem zurück"""
        system = platform.system().lower()
        arch = platform.machine().lower()
        
        if system == "windows":
            if "64" in arch or "amd64" in arch:
                return "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
            else:
                return "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win32-gpl.zip"
        elif system == "darwin":  # macOS
            return "https://evermeet.cx/ffmpeg/ffmpeg-6.0.zip"
        else:  # Linux
            # Für Linux nehmen wir an, dass FFmpeg über Package Manager installiert wird
            raise RuntimeError("Linux: FFmpeg muss über Package Manager installiert werden")
    
    def _download_ffmpeg(self) -> Path:
        """Lädt FFmpeg herunter und gibt den Pfad zur ZIP-Datei zurück"""
        url = self._get_download_url()
        logger.info("Lade FFmpeg herunter von: %s", url)
        
        # Temporäre Datei für Download
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        temp_file.close()
        
        try:
            urlretrieve(url, temp_file.name)
            logger.info("FFmpeg erfolgreich heruntergeladen")
            return Path(temp_file.name)
        except URLError as e:
            logger.error("Download fehlgeschlagen: %s", e)
            raise RuntimeError(f"FFmpeg Download fehlgeschlagen: {e}")
    
    def _extract_ffmpeg(self, zip_path: Path) -> None:
        """Extrahiert FFmpeg aus der ZIP-Datei"""
        logger.info("Extrahiere FFmpeg nach: %s", self.ffmpeg_dir)
        
        # FFmpeg-Verzeichnis erstellen
        self.ffmpeg_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.ffmpeg_dir)
            
            # Unterverzeichnis finden und Inhalt eine Ebene höher verschieben
            extracted_dirs = [d for d in self.ffmpeg_dir.iterdir() if d.is_dir()]
            if extracted_dirs:
                # Nehme das erste gefundene Verzeichnis (sollte ffmpeg-* sein)
                source_dir = extracted_dirs[0]
                
                # Verschiebe Inhalt von source_dir nach ffmpeg_dir
                for item in source_dir.iterdir():
                    dest = self.ffmpeg_dir / item.name
                    if item.is_dir():
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.move(str(item), str(dest))
                    else:
                        shutil.move(str(item), str(dest))
                
                # Leeres source_dir löschen
                source_dir.rmdir()
            
            logger.info("FFmpeg erfolgreich extrahiert")
            
        except zipfile.BadZipFile as e:
            logger.error("ZIP-Datei beschädigt: %s", e)
            raise RuntimeError("FFmpeg ZIP-Datei ist beschädigt")
        except Exception as e:
            logger.error("Extraktion fehlgeschlagen: %s", e)
            raise RuntimeError(f"FFmpeg Extraktion fehlgeschlagen: {e}")
    
    def _test_ffmpeg(self) -> bool:
        """Testet ob FFmpeg funktioniert"""
        try:
            result = subprocess.run(
                [str(self.ffmpeg_exe), "-version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info("FFmpeg Test erfolgreich")
                return True
            else:
                logger.error("FFmpeg Test fehlgeschlagen: %s", result.stderr)
                return False
        except Exception as e:
            logger.error("FFmpeg Test fehlgeschlagen: %s", e)
            return False
    
    def install_ffmpeg(self) -> bool:
        """Installiert FFmpeg automatisch"""
        try:
            logger.info("Starte automatische FFmpeg-Installation...")
            
            # Download
            zip_path = self._download_ffmpeg()
            
            try:
                # Extraktion
                self._extract_ffmpeg(zip_path)
                
                # Test
                if self._test_ffmpeg():
                    logger.info("FFmpeg erfolgreich installiert")
                    return True
                else:
                    logger.error("FFmpeg-Installation fehlgeschlagen: Test nicht bestanden")
                    return False
                    
            finally:
                # ZIP-Datei löschen
                zip_path.unlink(missing_ok=True)
                
        except Exception as e:
            logger.error("FFmpeg-Installation fehlgeschlagen: %s", e)
            return False
    
    def ensure_ffmpeg(self) -> bool:
        """Stellt sicher, dass FFmpeg verfügbar ist. Installiert es bei Bedarf."""
        if self.is_ffmpeg_available():
            return True
        
        logger.info("FFmpeg nicht verfügbar, starte automatische Installation...")
        return self.install_ffmpeg()


# Globale Instanz
_ffmpeg_manager = None

def get_ffmpeg_manager() -> FFmpegManager:
    """Gibt die globale FFmpegManager-Instanz zurück"""
    global _ffmpeg_manager
    if _ffmpeg_manager is None:
        _ffmpeg_manager = FFmpegManager()
    return _ffmpeg_manager

def ensure_ffmpeg() -> bool:
    """Stellt sicher, dass FFmpeg verfügbar ist"""
    return get_ffmpeg_manager().ensure_ffmpeg()

def get_ffmpeg_path() -> str:
    """Gibt den Pfad zur FFmpeg-Executable zurück"""
    return get_ffmpeg_manager().get_ffmpeg_path()
