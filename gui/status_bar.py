"""Bottom status bar + collapsible log."""
from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class StatusDock(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._status_dot = QLabel("● Ready")
        self._status_dot.setStyleSheet("color: #00FF88;")
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._meta = QLabel("Total inference: — | GPU: —")
        self._meta.setStyleSheet("color: #9AA0A6; font-size: 11px;")
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(120)
        self._log.setVisible(False)
        self._toggle = QToolButton()
        self._toggle.setText("Show log ▲")
        self._toggle.setStyleSheet("color: #00D4FF; border: none;")
        self._toggle.clicked.connect(self._on_toggle)

        top = QHBoxLayout()
        top.addWidget(self._status_dot)
        top.addWidget(self._progress, stretch=1)
        top.addWidget(self._meta)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 4)
        root.addLayout(top)
        root.addWidget(self._toggle)
        root.addWidget(self._log)

    def _on_toggle(self) -> None:
        vis = not self._log.isVisible()
        self._log.setVisible(vis)
        self._toggle.setText("Hide log ▼" if vis else "Show log ▲")

    def set_state(self, mode: str) -> None:
        colors = {
            "ready": ("#00FF88", "● Ready"),
            "processing": ("#00D4FF", "● Processing"),
            "complete": ("#00FF88", "● Complete"),
            "error": ("#FF6B35", "● Error"),
        }
        c, txt = colors.get(mode, ("#9AA0A6", "●"))
        self._status_dot.setStyleSheet(f"color: {c};")
        self._status_dot.setText(txt)

    def set_progress(self, value: int, text: str = "") -> None:
        self._progress.setValue(max(0, min(100, value)))
        if text:
            self._progress.setFormat(text)
        else:
            self._progress.setFormat("%p%")

    def set_timing_line(self, line: str) -> None:
        self._meta.setText(line)

    def append_log(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.append(f"[{ts}] {message}")
