"""
Dark scientific theme QSS for PanoCount Analyzer.
Colors: bg #0A0E1A, accent cyan #00D4FF, purple #7B2FFF, success #00FF88, warning #FF6B35
"""

APP_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0A0E1A;
    color: #E8EAED;
    font-family: "Segoe UI", "Roboto", sans-serif;
    font-size: 13px;
}

QScrollArea { border: none; background: transparent; }

QGroupBox {
    border: 1px solid rgba(0, 212, 255, 0.35);
    border-radius: 8px;
    margin-top: 12px;
    padding: 12px 8px 8px 8px;
    background-color: rgba(26, 31, 58, 0.55);
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #00D4FF;
    font-weight: bold;
    font-size: 14px;
}

QLabel { color: #E8EAED; }
QLabel[class="subtitle"] { color: #9AA0A6; font-size: 11px; }
QLabel[class="badge"] {
    background-color: rgba(123, 47, 255, 0.25);
    border: 1px solid #7B2FFF;
    border-radius: 10px;
    padding: 4px 10px;
    color: #E8EAED;
    font-size: 11px;
}
QLabel[class="badge-green"] {
    background-color: rgba(0, 255, 136, 0.15);
    border: 1px solid #00FF88;
    color: #00FF88;
}
QLabel[class="badge-orange"] {
    background-color: rgba(255, 107, 53, 0.15);
    border: 1px solid #FF6B35;
    color: #FF6B35;
}
QLabel[class="badge-grey"] {
    background-color: rgba(255,255,255,0.06);
    border: 1px solid #5F6368;
    color: #B0B3B8;
}

QPushButton {
    background-color: rgba(0, 212, 255, 0.12);
    border: 1px solid rgba(0, 212, 255, 0.5);
    border-radius: 8px;
    padding: 8px 14px;
    color: #00D4FF;
    font-weight: 600;
}
QPushButton:hover {
    background-color: rgba(0, 212, 255, 0.22);
    border: 1px solid #00D4FF;
}
QPushButton:pressed { background-color: rgba(0, 212, 255, 0.08); }
QPushButton[class="primary"] {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #00D4FF, stop:1 #7B2FFF);
    color: #0A0E1A;
    border: 1px solid #00D4FF;
    font-size: 15px;
    padding: 12px;
}
QPushButton[class="export-json"] { border-color: #00FF88; color: #00FF88; }
QPushButton[class="export-img"] { border-color: #7B2FFF; color: #C9A0FF; }
QPushButton[class="export-clip"] { border-color: #00D4FF; color: #00D4FF; }

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: rgba(10, 14, 26, 0.9);
    border: 1px solid rgba(0, 212, 255, 0.25);
    border-radius: 6px;
    padding: 6px 8px;
    color: #E8EAED;
    selection-background-color: rgba(0, 212, 255, 0.35);
}
QComboBox::drop-down { border: none; width: 24px; }

QSlider::groove:horizontal {
    height: 6px;
    background: rgba(255,255,255,0.08);
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #00D4FF, stop:1 #7B2FFF);
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
    border: 1px solid #00D4FF;
}

QCheckBox { spacing: 8px; color: #E8EAED; }
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid rgba(0, 212, 255, 0.45);
    background: rgba(10, 14, 26, 0.9);
}
QCheckBox::indicator:checked {
    background: rgba(0, 212, 255, 0.35);
    border: 1px solid #00D4FF;
}

QProgressBar {
    border: 1px solid rgba(0, 212, 255, 0.35);
    border-radius: 6px;
    text-align: center;
    color: #E8EAED;
    background: rgba(10, 14, 26, 0.9);
    height: 22px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #00D4FF, stop:1 #7B2FFF);
    border-radius: 5px;
}

QTabWidget::pane {
    border: 1px solid rgba(0, 212, 255, 0.25);
    border-radius: 8px;
    top: -1px;
    background: rgba(26, 31, 58, 0.45);
}
QTabBar::tab {
    background: rgba(10, 14, 26, 0.7);
    color: #9AA0A6;
    padding: 10px 16px;
    margin-right: 4px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    border: 1px solid rgba(255,255,255,0.06);
}
QTabBar::tab:selected {
    color: #00D4FF;
    border: 1px solid rgba(0, 212, 255, 0.55);
    background: rgba(0, 212, 255, 0.12);
}

QTableWidget {
    gridline-color: rgba(0, 212, 255, 0.15);
    background: rgba(10, 14, 26, 0.65);
    alternate-background-color: rgba(26, 31, 58, 0.35);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 8px;
}
QHeaderView::section {
    background: rgba(0, 212, 255, 0.12);
    color: #00D4FF;
    padding: 8px;
    border: none;
    font-weight: bold;
}

QTextEdit {
    background: #050810;
    color: #C4C7CE;
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 6px;
    font-family: Consolas, "Courier New", monospace;
    font-size: 11px;
}

QFrame[class="glow-panel"] {
    border: 1px solid rgba(0, 212, 255, 0.35);
    border-radius: 8px;
    background: rgba(26, 31, 58, 0.4);
}
"""

HEADER_STYLE = """
QFrame#headerBar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #0A0E1A, stop:1 #1A1F3A);
    border: none;
    border-bottom: 2px solid rgba(0, 212, 255, 0.45);
    min-height: 64px;
}
"""
