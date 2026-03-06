# STTDesktop - Speech-to-Text Desktop-Anwendung

Desktop-Anwendung für Speech-to-Text-Transkription mit Groq API (Whisper), KI-basierter Grammatikkorrektur, Übersetzung und Dialog-Modus mit Sprachausgabe.

[![Latest Release](https://img.shields.io/github/v/release/Dolcruz/Speech-to-text-desktop?label=Download)](https://github.com/Dolcruz/Speech-to-text-desktop/releases/latest)

## Download

**Windows-Executable:** [STTDesktop.exe](https://github.com/Dolcruz/Speech-to-text-desktop/releases/latest/download/STTDesktop.exe)

Die .exe benötigt keine Installation. Bei Windows Defender-Warnung auf "Weitere Informationen" und "Trotzdem ausführen" klicken.

## Funktionen

### Kernfunktionen
- Speech-to-Text-Transkription über Groq API (Whisper-Modell)
- Grammatikkorrektur mittels LLM (Kimi K2)
- Übersetzung in mehrere Sprachen
- Dialog-Modus mit bidirektionaler Übersetzung und Sprachausgabe
- Verlauf mit Volltext-Ansicht und Nachbearbeitung
- Automatisches Update-System über GitHub Releases

### Benutzeroberfläche
- Dark Mode Design
- Organische Sphere-Visualisierung mit Echtzeit-Audio-Feedback
- Anpassbare Visualisierung (Detailgrad, Glow, Farbe)
- Globale Hotkeys (Alt+T, Esc) - erfordert Administratorrechte
- Lokale Tastenkombinationen (funktionieren immer bei Fokus)

### Audio
- Mikrofon-Auswahl
- Echtzeit-Audio-Pegel-Visualisierung
- Konfigurierbare Aufnahmedauer und Stille-Erkennung

## Erste Schritte

### 1. API-Key einrichten

Groq API Key (kostenlos) ist erforderlich:

1. Account erstellen: https://console.groq.com/keys
2. API Key kopieren
3. In Einstellungen unter "Groq API Key" einfügen

### 2. Verwendung

**Standard-Modus:**
- Alt+T oder "Aufnahme starten" zum Starten
- Alt+T erneut oder "Aufnahme beenden" zum Stoppen
- Transkription erscheint im Ergebnisfenster
- Optional: Grammatikkorrektur und Übersetzung anwenden

**Dialog-Modus:**
- "Dialog-Modus" öffnen
- Zwei Sprachen auswählen
- Abwechselnd in beiden Sprachen sprechen
- Übersetzung wird automatisch vorgelesen

**Verfügbare Sprachen:**
- Deutsch
- Englisch
- Spanisch
- Französisch
- Italienisch
- Arabisch

### 3. Globale Hotkeys aktivieren

Globale Hotkeys (systemweit funktionierend) benötigen Administratorrechte:

**Für .exe-Version:**
1. Rechtsklick auf `STTDesktop.exe`
2. "Als Administrator ausführen"

**Oder:** Verwende `STTDesktop_Admin.bat` (liegt neben der .exe)

Ohne Admin-Rechte funktionieren die Hotkeys nur, wenn das Fenster im Fokus ist.

## Einstellungen

### Hotkeys
- **Alt+T**: Aufnahme starten/stoppen
- **Esc**: Aufnahme abbrechen

### Audio
- Mikrofon-Gerät auswählen
- Maximale Aufnahmedauer (0 = unbegrenzt)
- Stille-Schwelle (RMS-Wert)

### Verhalten
- Automatisch kopieren (nach Transkription)
- Automatisch einfügen (nach Transkription)
- Grammatik automatisch korrigieren

### Visualisierung
- Detailgrad (200-1000)
- Glow-Intensität (0.1-2.0)
- Farbe (Hue 0-359)
- Live-Vorschau mit Mikrofon-Input

## Entwicklung

### Voraussetzungen
- Python 3.10+
- Windows 10/11

### Installation

```bash
git clone https://github.com/Dolcruz/Speech-to-text-desktop.git
cd Speech-to-text-desktop

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt

python main.py
```

### .exe bauen

```powershell
.\build_release.ps1
```

Oder manuell:

```powershell
pip install pyinstaller
.\build_windows.ps1 -OneFile
```

Resultat: `dist\STTDesktop.exe`

### Release erstellen

Siehe [RELEASE_GUIDE.md](RELEASE_GUIDE.md) für Details.

Kurzversion:
1. VERSION-Datei aktualisieren (z.B. `1.3.0`)
2. .exe bauen mit `.\build_release.ps1`
3. GitHub Release erstellen mit Tag `v1.3.0`
4. `STTDesktop.exe` als Asset hochladen

Das Auto-Update-System erkennt neue Releases automatisch.

## Architektur

```
main.py                    # Hauptcontroller, Logik
├── stt_app/
│   ├── ui_main.py        # Hauptfenster
│   ├── ui_overlay.py     # Visualisierungs-Overlay
│   ├── ui_dialog.py      # Dialog-Modus
│   ├── ui_update.py      # Update-Dialog
│   ├── ui_result_popup.py # Ergebnis-Anzeige
│   ├── audio.py          # Audio-Aufnahme
│   ├── groq_client.py    # Groq API Client
│   ├── tts_client.py     # Text-to-Speech (Edge TTS)
│   ├── updater.py        # Update-System
│   ├── hotkeys.py        # Globale Hotkeys
│   ├── theme.py          # UI Theme
│   ├── config.py         # Konfiguration
│   └── logger.py         # Logging
```

## Technischer Stack

- **UI**: PySide6 (Qt for Python)
- **STT**: Groq API (Whisper large-v3-turbo)
- **LLM**: Groq API (moonshotai/kimi-k2-instruct)
- **TTS**: Microsoft Edge TTS (kostenlos)
- **Audio**: sounddevice, soundfile, numpy
- **Hotkeys**: keyboard (globale Hotkeys)
- **Updates**: GitHub Releases API, requests
- **Packaging**: PyInstaller

## API-Kosten

- **Groq API**: Kostenloser Tier verfügbar (Rate Limits gelten)
- **Edge TTS**: Komplett kostenlos, keine API-Keys erforderlich
- **GitHub Releases**: Kostenlos für öffentliche Repositories

## Lizenz

MIT License - siehe [LICENSE](LICENSE)

## Credits

- OpenAI Whisper
- Groq für API-Zugang
- Microsoft Edge TTS
- PySide6/Qt Framework

