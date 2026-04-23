"""Animation helpers for PanoCount Analyzer."""
from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QObject
from PySide6.QtWidgets import QWidget, QGraphicsOpacityEffect


def fade_in_widget(widget: QWidget, duration_ms: int = 500) -> None:
    eff = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(eff)
    anim = QPropertyAnimation(eff, b"opacity")
    anim.setDuration(duration_ms)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
    anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)


def pulse_opacity(target: QObject, prop: bytes, min_v: float, max_v: float, duration_ms: int = 800):
    """Returns a looping pulse animation (caller must keep reference)."""
    anim = QPropertyAnimation(target, prop)
    anim.setDuration(duration_ms)
    anim.setStartValue(min_v)
    anim.setEndValue(max_v)
    anim.setEasingCurve(QEasingCurve.Type.InOutSine)
    anim.setLoopCount(-1)
    return anim
