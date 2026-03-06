from __future__ import annotations

import math
import random
from dataclasses import dataclass

from PySide6 import QtCore, QtGui, QtWidgets


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class FlowFieldTerm:
    """Single sine term used to fake organic surface flow."""

    frequency: float
    amplitude: float
    speed: float
    phase: float


class OrganicSphereWidget(QtWidgets.QWidget):
    """Procedural recording visualization with a soft, audio-reactive blob."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._detail_setting = 600
        self._sample_count = 96
        self._glow_intensity = 1.0
        self._color_hue = 200
        self._target_level = 0.0
        self._current_level = 0.0
        self._smoothed_energy = 0.0
        self._time_seconds = 0.0
        self._angles: list[float] = []

        rng = random.Random(31)
        self._macro_terms = (
            FlowFieldTerm(2.0, 0.050, 0.42, rng.uniform(0.0, math.tau)),
            FlowFieldTerm(3.0, 0.034, -0.30, rng.uniform(0.0, math.tau)),
            FlowFieldTerm(5.0, 0.017, 0.74, rng.uniform(0.0, math.tau)),
        )
        self._surface_terms = (
            FlowFieldTerm(7.0, 0.010, 1.08, rng.uniform(0.0, math.tau)),
            FlowFieldTerm(11.0, 0.007, -0.92, rng.uniform(0.0, math.tau)),
            FlowFieldTerm(15.0, 0.004, 1.46, rng.uniform(0.0, math.tau)),
        )
        self._audio_terms = (
            FlowFieldTerm(2.0, 0.040, 1.82, rng.uniform(0.0, math.tau)),
            FlowFieldTerm(4.0, 0.022, -1.28, rng.uniform(0.0, math.tau)),
            FlowFieldTerm(6.0, 0.010, 2.30, rng.uniform(0.0, math.tau)),
        )
        self._accent_phases = tuple(rng.uniform(0.0, math.tau) for _ in range(3))

        self._animation_timer = QtCore.QTimer(self)
        self._animation_timer.timeout.connect(self._animate)
        self._animation_timer.start(16)

        self.set_particle_count(self._detail_setting)
        self.setMinimumSize(360, 360)

    @QtCore.Slot(float)
    def set_level(self, level: float) -> None:
        """Set normalized audio level with a little extra gain for visibility."""
        self._target_level = _clamp(level * 6.0, 0.0, 1.0)

    @QtCore.Slot(int)
    def set_particle_count(self, count: int) -> None:
        """Reuse the persisted detail setting as contour resolution."""
        self._detail_setting = max(200, min(1000, count))
        detail_ratio = (self._detail_setting - 200) / 800.0
        self._sample_count = 56 + int(round(detail_ratio * 88))
        self._angles = [index * math.tau / self._sample_count for index in range(self._sample_count)]
        self.update()

    @QtCore.Slot(float)
    def set_glow_intensity(self, intensity: float) -> None:
        """Set glow intensity (0.0 to 2.0)."""
        self._glow_intensity = _clamp(intensity, 0.0, 2.0)
        self.update()

    @QtCore.Slot(int)
    def set_color_hue(self, hue: int) -> None:
        """Set hue (0-359)."""
        self._color_hue = max(0, min(359, hue))
        self.update()

    def _animate(self) -> None:
        """Advance animation time and smooth audio response."""
        smoothing = 0.24 if self._target_level > self._current_level else 0.10
        self._current_level += (self._target_level - self._current_level) * smoothing
        self._smoothed_energy += (self._current_level - self._smoothed_energy) * 0.08
        self._time_seconds += 0.016
        self.update()

    def _field_value(self, angle: float, terms: tuple[FlowFieldTerm, ...]) -> float:
        value = 0.0
        for term in terms:
            drift = 0.24 * math.sin(self._time_seconds * (term.speed * 0.37) + term.phase * 0.6)
            value += term.amplitude * math.sin(
                (angle + drift) * term.frequency + self._time_seconds * term.speed + term.phase
            )
        return value

    def _build_blob_points(self, center: QtCore.QPointF, radius: float) -> list[QtCore.QPointF]:
        points: list[QtCore.QPointF] = []
        breathing = 1.0 + 0.018 * math.sin(self._time_seconds * 0.68)
        audio_scale = 1.0 + 0.12 * self._smoothed_energy
        shimmer_strength = 0.55 + 0.45 * self._smoothed_energy
        pulse_strength = 0.65 * self._current_level

        for angle in self._angles:
            macro = self._field_value(angle, self._macro_terms)
            surface_angle = angle + 0.18 * math.sin(angle * 2.0 - self._time_seconds * 0.46)
            shimmer = self._field_value(surface_angle, self._surface_terms) * shimmer_strength
            pulse = self._field_value(angle - self._time_seconds * 0.25, self._audio_terms) * pulse_strength

            # Clamp the contour so the blob stays smooth and never self-intersects.
            contour = _clamp(1.0 + macro + shimmer + pulse, 0.78, 1.28)
            distance = radius * breathing * audio_scale * contour
            points.append(
                QtCore.QPointF(
                    center.x() + math.cos(angle) * distance,
                    center.y() + math.sin(angle) * distance,
                )
            )

        return points

    @staticmethod
    def _midpoint(first: QtCore.QPointF, second: QtCore.QPointF) -> QtCore.QPointF:
        return QtCore.QPointF((first.x() + second.x()) * 0.5, (first.y() + second.y()) * 0.5)

    def _build_smooth_path(self, points: list[QtCore.QPointF]) -> QtGui.QPainterPath:
        path = QtGui.QPainterPath()
        if not points:
            return path

        start = self._midpoint(points[-1], points[0])
        path.moveTo(start)
        for index, point in enumerate(points):
            next_point = points[(index + 1) % len(points)]
            path.quadTo(point, self._midpoint(point, next_point))
        path.closeSubpath()
        return path

    def _draw_detached_accents(
        self,
        painter: QtGui.QPainter,
        center: QtCore.QPointF,
        radius: float,
        glow_color: QtGui.QColor,
    ) -> None:
        accent_boost = 0.55 + 0.45 * self._glow_intensity
        for index, phase in enumerate(self._accent_phases):
            orbit_angle = self._time_seconds * (0.32 + index * 0.09) + phase
            orbit_radius = radius * (1.08 + 0.04 * math.sin(self._time_seconds * 0.5 + phase))
            point = QtCore.QPointF(
                center.x() + math.cos(orbit_angle) * orbit_radius,
                center.y() + math.sin(orbit_angle * 1.25) * radius * 0.90,
            )
            size = radius * (0.030 + 0.014 * self._smoothed_energy + index * 0.003)
            gradient = QtGui.QRadialGradient(point, size * 2.5)
            alpha = int((28 + index * 12 + 34 * self._smoothed_energy) * accent_boost)
            gradient.setColorAt(0.0, QtGui.QColor(glow_color.red(), glow_color.green(), glow_color.blue(), alpha))
            gradient.setColorAt(0.45, QtGui.QColor(glow_color.red(), glow_color.green(), glow_color.blue(), alpha // 3))
            gradient.setColorAt(1.0, QtGui.QColor(0, 0, 0, 0))

            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(gradient)
            painter.drawEllipse(point, size * 2.3, size * 2.3)

    def _draw_glow_outline(
        self,
        painter: QtGui.QPainter,
        path: QtGui.QPainterPath,
        glow_color: QtGui.QColor,
        radius: float,
    ) -> None:
        glow_scale = 0.55 + 0.45 * self._smoothed_energy
        for width_factor, alpha in ((0.26, 20), (0.16, 42), (0.09, 84)):
            pen = QtGui.QPen(
                QtGui.QColor(
                    glow_color.red(),
                    glow_color.green(),
                    glow_color.blue(),
                    int(alpha * (0.45 + self._glow_intensity * 0.75) * (0.80 + glow_scale * 0.35)),
                ),
                max(2.0, radius * width_factor),
                QtCore.Qt.SolidLine,
                QtCore.Qt.RoundCap,
                QtCore.Qt.RoundJoin,
            )
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawPath(path)

    def _draw_inner_surface(
        self,
        painter: QtGui.QPainter,
        blob_path: QtGui.QPainterPath,
        center: QtCore.QPointF,
        radius: float,
    ) -> None:
        hue = self._color_hue
        painter.save()
        painter.setClipPath(blob_path)

        internal_glow = QtGui.QRadialGradient(
            QtCore.QPointF(
                center.x() + math.cos(self._time_seconds * 0.72) * radius * 0.10,
                center.y() - radius * 0.14 + math.sin(self._time_seconds * 0.58) * radius * 0.06,
            ),
            radius * 0.92,
        )
        internal_glow.setColorAt(0.0, QtGui.QColor.fromHsv(hue, 60, 255, int(130 + 70 * self._smoothed_energy)))
        internal_glow.setColorAt(0.4, QtGui.QColor.fromHsv(hue, 85, 255, int(75 + 30 * self._smoothed_energy)))
        internal_glow.setColorAt(1.0, QtGui.QColor.fromHsv(hue, 140, 80, 0))
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(internal_glow)
        painter.drawEllipse(center, radius * 0.96, radius * 0.96)

        shadow_gradient = QtGui.QLinearGradient(
            QtCore.QPointF(center.x(), center.y() - radius * 0.2),
            QtCore.QPointF(center.x(), center.y() + radius),
        )
        shadow_gradient.setColorAt(0.0, QtGui.QColor(0, 0, 0, 0))
        shadow_gradient.setColorAt(1.0, QtGui.QColor(4, 9, 16, 110))
        painter.fillRect(blob_path.boundingRect(), shadow_gradient)

        highlight_center = QtCore.QPointF(center.x() - radius * 0.28, center.y() - radius * 0.30)
        highlight = QtGui.QRadialGradient(highlight_center, radius * 0.36)
        highlight.setColorAt(0.0, QtGui.QColor(255, 255, 255, int(105 + 40 * self._smoothed_energy)))
        highlight.setColorAt(0.65, QtGui.QColor.fromHsv(hue, 35, 255, 30))
        highlight.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        painter.setBrush(highlight)
        painter.drawEllipse(highlight_center, radius * 0.34, radius * 0.34)

        filament_color = QtGui.QColor.fromHsv(hue, 70, 255, int(42 + 28 * self._smoothed_energy))
        for index, phase in enumerate(self._accent_phases[:2]):
            band_y = center.y() + math.sin(self._time_seconds * (0.78 + index * 0.18) + phase) * radius * (0.11 + 0.03 * index)
            spread = radius * (0.14 + 0.05 * index)
            ribbon = QtGui.QPainterPath(QtCore.QPointF(center.x() - radius * 0.72, band_y))
            ribbon.cubicTo(
                QtCore.QPointF(center.x() - radius * 0.30, band_y - spread),
                QtCore.QPointF(center.x() + radius * 0.08, band_y + spread * 0.82),
                QtCore.QPointF(center.x() + radius * 0.72, band_y - spread * 0.08),
            )
            pen = QtGui.QPen(
                filament_color,
                max(1.4, radius * (0.020 - index * 0.004)),
                QtCore.Qt.SolidLine,
                QtCore.Qt.RoundCap,
                QtCore.Qt.RoundJoin,
            )
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawPath(ribbon)

        painter.restore()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        if not self._angles:
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        center = QtCore.QPointF(self.width() * 0.5, self.height() * 0.5)
        radius = min(self.width(), self.height()) * 0.28
        blob_path = self._build_smooth_path(self._build_blob_points(center, radius))

        hue = self._color_hue
        glow_color = QtGui.QColor.fromHsv(hue, 175, 255)
        rim_light = QtGui.QColor.fromHsv(hue, 45, 255, 180)
        rim_shadow = QtGui.QColor.fromHsv(hue, 185, 160, 50)

        aura_radius = radius * (1.55 + 0.18 * self._smoothed_energy) * (0.78 + 0.22 * self._glow_intensity)
        aura = QtGui.QRadialGradient(center, aura_radius)
        aura_alpha = int(72 * (0.35 + self._glow_intensity * 0.65) * (0.75 + self._smoothed_energy * 0.25))
        aura.setColorAt(0.0, QtGui.QColor(glow_color.red(), glow_color.green(), glow_color.blue(), aura_alpha))
        aura.setColorAt(0.45, QtGui.QColor(glow_color.red(), glow_color.green(), glow_color.blue(), aura_alpha // 2))
        aura.setColorAt(1.0, QtGui.QColor(0, 0, 0, 0))

        painter.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(aura)
        painter.drawEllipse(center, aura_radius, aura_radius)
        self._draw_glow_outline(painter, blob_path, glow_color, radius)
        self._draw_detached_accents(painter, center, radius, glow_color)

        painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)

        fill_gradient = QtGui.QRadialGradient(
            QtCore.QPointF(
                center.x() - radius * 0.18 + math.cos(self._time_seconds * 0.66) * radius * 0.03,
                center.y() - radius * 0.24 + math.sin(self._time_seconds * 0.52) * radius * 0.04,
            ),
            radius * 1.34,
        )
        fill_gradient.setFocalPoint(center.x() - radius * 0.28, center.y() - radius * 0.32)
        fill_gradient.setColorAt(0.0, QtGui.QColor.fromHsv(hue, 70, 255, 246))
        fill_gradient.setColorAt(0.42, QtGui.QColor.fromHsv(hue, 145, 225, 236))
        fill_gradient.setColorAt(1.0, QtGui.QColor.fromHsv(hue, 190, 88, 232))
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(fill_gradient)
        painter.drawPath(blob_path)

        self._draw_inner_surface(painter, blob_path, center, radius)

        edge_gradient = QtGui.QLinearGradient(
            QtCore.QPointF(center.x() - radius, center.y() - radius),
            QtCore.QPointF(center.x() + radius, center.y() + radius),
        )
        edge_gradient.setColorAt(0.0, rim_light)
        edge_gradient.setColorAt(0.45, QtGui.QColor.fromHsv(hue, 70, 255, 110))
        edge_gradient.setColorAt(1.0, rim_shadow)
        edge_pen = QtGui.QPen(
            QtGui.QBrush(edge_gradient),
            max(1.8, radius * 0.028),
            QtCore.Qt.SolidLine,
            QtCore.Qt.RoundCap,
            QtCore.Qt.RoundJoin,
        )
        painter.setPen(edge_pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawPath(blob_path)


class RecordingOverlay(QtWidgets.QWidget):
    """Frameless, always-on-top overlay showing recording status."""

    cancel_requested = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)

        self._elapsed = 0.0
        self._level = 0.0

        container = QtWidgets.QFrame()
        container.setStyleSheet(
            """
            QFrame {
                background-color: rgba(18, 18, 18, 250);
                border-radius: 20px;
                border: 2px solid rgba(80, 80, 80, 180);
            }
            """
        )

        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(40)
        shadow.setOffset(0, 8)
        shadow.setColor(QtGui.QColor(0, 0, 0, 150))
        container.setGraphicsEffect(shadow)

        self._visualization = OrganicSphereWidget()
        self._visualization.setMinimumSize(360, 360)

        rec_label = QtWidgets.QLabel("AUFNAHME")
        rec_label.setAlignment(QtCore.Qt.AlignCenter)
        rec_label.setStyleSheet(
            """
            color: #ff4444;
            font-size: 11pt;
            font-weight: 700;
            letter-spacing: 2px;
            background: transparent;
            """
        )

        self._timer_label = QtWidgets.QLabel("00:00")
        self._timer_label.setAlignment(QtCore.Qt.AlignCenter)
        self._timer_label.setStyleSheet(
            """
            color: #ffffff;
            font-size: 28pt;
            font-weight: 700;
            background: transparent;
            letter-spacing: 2px;
            """
        )

        cancel_btn = QtWidgets.QPushButton("Abbrechen")
        cancel_btn.setFixedSize(120, 38)
        cancel_btn.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(100, 50, 50, 200);
                border: 1.5px solid rgba(150, 70, 70, 255);
                border-radius: 19px;
                color: #ff6b6b;
                font-size: 10pt;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: rgba(120, 60, 60, 255);
                border-color: rgba(180, 90, 90, 255);
            }
            """
        )
        cancel_btn.clicked.connect(self.cancel_requested.emit)

        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(20, 20, 20, 24)
        container_layout.setSpacing(12)
        container_layout.addWidget(rec_label, alignment=QtCore.Qt.AlignCenter)
        container_layout.addWidget(self._visualization, alignment=QtCore.Qt.AlignCenter)
        container_layout.addWidget(self._timer_label, alignment=QtCore.Qt.AlignCenter)
        container_layout.addWidget(cancel_btn, alignment=QtCore.Qt.AlignCenter)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(container)

        self.setFixedSize(420, 540)

    @QtCore.Slot(int)
    def set_particle_count(self, count: int) -> None:
        self._visualization.set_particle_count(count)

    @QtCore.Slot(float)
    def set_glow_intensity(self, intensity: float) -> None:
        self._visualization.set_glow_intensity(intensity)

    @QtCore.Slot(int)
    def set_color_hue(self, hue: int) -> None:
        self._visualization.set_color_hue(hue)

    @QtCore.Slot(float)
    def update_level(self, level: float) -> None:
        """Update the audio level for the organic sphere."""
        self._level = level
        self._visualization.set_level(level)

    @QtCore.Slot(float)
    def update_time(self, seconds: float) -> None:
        """Update the elapsed time display."""
        self._elapsed = seconds
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        self._timer_label.setText(f"{mins:02d}:{secs:02d}")

    def show_top_right(self) -> None:
        """Show the overlay at the top-right of the screen."""
        screen = QtWidgets.QApplication.primaryScreen()
        if not screen:
            self.show()
            return

        geo = screen.availableGeometry()
        margin = 20
        x = geo.x() + geo.width() - self.width() - margin
        y = geo.y() + margin

        self.move(x, y)
        self.show()
        self.raise_()
        self.activateWindow()

