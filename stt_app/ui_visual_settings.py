from __future__ import annotations

from typing import Optional
from PySide6 import QtCore, QtWidgets


class VisualSettingsDialog(QtWidgets.QDialog):
    """Dialog for configuring the recording visualization with live preview."""
    
    # Signals for live updates
    particle_count_changed = QtCore.Signal(int)
    glow_intensity_changed = QtCore.Signal(float)
    color_hue_changed = QtCore.Signal(int)
    
    def __init__(self, particle_count: int = 600, glow_intensity: float = 1.0, 
                 color_hue: int = 200, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Visualisierungs-Einstellungen")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        
        # Store initial values
        self._initial_count = particle_count
        self._initial_glow = glow_intensity
        self._initial_hue = color_hue
        
        # Dark mode styling
        self.setStyleSheet("""
            QDialog {
                background-color: #181818;
            }
        """)
        
        # Title
        title = QtWidgets.QLabel("Sphere-Visualisierung anpassen")
        title.setStyleSheet("""
            font-size: 16pt;
            font-weight: 700;
            color: #ffffff;
            padding: 8px 0px;
        """)
        
        # Info label
        info = QtWidgets.QLabel("Detailgrad, Glow und Farbe werden live angezeigt. Starte eine Aufnahme fuer Echtzeit-Preview.")
        info.setStyleSheet("""
            font-size: 9pt;
            color: #a0a0a0;
            padding: 4px 0px 16px 0px;
        """)
        info.setWordWrap(True)
        
        # Detail Slider
        count_label = QtWidgets.QLabel("Detailgrad:")
        count_label.setStyleSheet("font-size: 11pt; font-weight: 600; color: #ffffff;")
        
        self._count_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._count_slider.setMinimum(200)
        self._count_slider.setMaximum(1000)
        self._count_slider.setValue(self._initial_count)
        self._count_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self._count_slider.setTickInterval(100)
        
        self._count_value_label = QtWidgets.QLabel(str(self._initial_count))
        self._count_value_label.setStyleSheet("font-size: 10pt; color: #4287f5; font-weight: 600; min-width: 60px;")
        
        count_layout = QtWidgets.QHBoxLayout()
        count_layout.addWidget(self._count_slider, 1)
        count_layout.addWidget(self._count_value_label)
        
        # Glow Intensity Slider
        glow_label = QtWidgets.QLabel("Glow-Intensität:")
        glow_label.setStyleSheet("font-size: 11pt; font-weight: 600; color: #ffffff;")
        
        self._glow_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._glow_slider.setMinimum(0)
        self._glow_slider.setMaximum(200)  # 0.0 to 2.0 (x100)
        self._glow_slider.setValue(int(self._initial_glow * 100))
        self._glow_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self._glow_slider.setTickInterval(25)
        
        self._glow_value_label = QtWidgets.QLabel(f"{self._initial_glow:.1f}")
        self._glow_value_label.setStyleSheet("font-size: 10pt; color: #4287f5; font-weight: 600; min-width: 60px;")
        
        glow_layout = QtWidgets.QHBoxLayout()
        glow_layout.addWidget(self._glow_slider, 1)
        glow_layout.addWidget(self._glow_value_label)
        
        # Color Hue Slider
        color_label = QtWidgets.QLabel("Farbe:")
        color_label.setStyleSheet("font-size: 11pt; font-weight: 600; color: #ffffff;")
        
        self._color_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._color_slider.setMinimum(0)
        self._color_slider.setMaximum(360)
        self._color_slider.setValue(self._initial_hue)
        self._color_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self._color_slider.setTickInterval(60)
        
        self._color_preview = QtWidgets.QLabel()
        self._color_preview.setFixedSize(60, 30)
        self._color_preview.setStyleSheet("background-color: #007aff; border-radius: 6px;")
        
        color_layout = QtWidgets.QHBoxLayout()
        color_layout.addWidget(self._color_slider, 1)
        color_layout.addWidget(self._color_preview)
        
        # Presets
        presets_label = QtWidgets.QLabel("Presets:")
        presets_label.setStyleSheet("font-size: 11pt; font-weight: 600; color: #ffffff; padding-top: 16px;")
        
        preset_blue_btn = QtWidgets.QPushButton("Neonblau")
        preset_cyan_btn = QtWidgets.QPushButton("Cyan")
        preset_purple_btn = QtWidgets.QPushButton("Lila")
        preset_green_btn = QtWidgets.QPushButton("Grün")
        preset_red_btn = QtWidgets.QPushButton("Rot")
        
        presets_layout = QtWidgets.QHBoxLayout()
        presets_layout.addWidget(preset_blue_btn)
        presets_layout.addWidget(preset_cyan_btn)
        presets_layout.addWidget(preset_purple_btn)
        presets_layout.addWidget(preset_green_btn)
        presets_layout.addWidget(preset_red_btn)
        
        # Buttons
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        
        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)
        layout.addWidget(title)
        layout.addWidget(info)
        layout.addWidget(count_label)
        layout.addLayout(count_layout)
        layout.addWidget(glow_label)
        layout.addLayout(glow_layout)
        layout.addWidget(color_label)
        layout.addLayout(color_layout)
        layout.addWidget(presets_label)
        layout.addLayout(presets_layout)
        layout.addStretch()
        layout.addWidget(btn_box)
        
        # Connect signals for live updates
        self._count_slider.valueChanged.connect(self._on_count_changed)
        self._glow_slider.valueChanged.connect(self._on_glow_changed)
        self._color_slider.valueChanged.connect(self._on_color_changed)
        
        # Connect presets
        preset_blue_btn.clicked.connect(lambda: self._apply_preset(200))
        preset_cyan_btn.clicked.connect(lambda: self._apply_preset(180))
        preset_purple_btn.clicked.connect(lambda: self._apply_preset(280))
        preset_green_btn.clicked.connect(lambda: self._apply_preset(120))
        preset_red_btn.clicked.connect(lambda: self._apply_preset(0))
        
        # Initialize color preview
        self._update_color_preview()
    
    def _on_count_changed(self, value: int) -> None:
        """Handle detail slider change."""
        self._count_value_label.setText(str(value))
        self.particle_count_changed.emit(value)
    
    def _on_glow_changed(self, value: int) -> None:
        """Handle glow intensity slider change."""
        intensity = value / 100.0  # Convert to 0.0-2.0 range
        self._glow_value_label.setText(f"{intensity:.1f}")
        self.glow_intensity_changed.emit(intensity)
    
    def _on_color_changed(self, value: int) -> None:
        """Handle color hue slider change."""
        self._update_color_preview()
        self.color_hue_changed.emit(value)
    
    def _update_color_preview(self) -> None:
        """Update the color preview box."""
        from PySide6.QtGui import QColor
        hue = self._color_slider.value()
        color = QColor.fromHsv(hue, 200, 255)
        self._color_preview.setStyleSheet(
            f"background-color: {color.name()}; border-radius: 6px;"
        )
    
    def _apply_preset(self, hue: int) -> None:
        """Apply a color preset."""
        self._color_slider.setValue(hue)
    
    def get_values(self) -> tuple[int, float, int]:
        """Get current slider values."""
        return (
            self._count_slider.value(),
            self._glow_slider.value() / 100.0,
            self._color_slider.value()
        )

