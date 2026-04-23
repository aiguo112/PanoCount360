#!/usr/bin/env python3
"""
PanoCount Analyzer — desktop GUI for panoramic crowd counting (PCP).

Run from project root:
    python main.py
"""
from __future__ import annotations

import os
import sys

# Windows: PyTorch/CUDA DLL paths before gui (and torch) import — avoids WinError 182 on nvfuser etc.
if sys.platform == "win32":
    import importlib.util

    conda_prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    dll_paths = [
        os.path.join(conda_prefix, "Library", "bin"),
        os.path.join(conda_prefix, "Library", "mingw-w64", "bin"),
        os.path.join(conda_prefix, "Library", "usr", "bin"),
        os.path.join(conda_prefix, "Scripts"),
    ]
    try:
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec and getattr(torch_spec, "origin", None):
            torch_lib = os.path.join(os.path.dirname(torch_spec.origin), "lib")
            if os.path.isdir(torch_lib):
                dll_paths.insert(0, torch_lib)
    except Exception:
        pass
    _torch_lib_fallback = os.path.join(conda_prefix, "Lib", "site-packages", "torch", "lib")
    if os.path.isdir(_torch_lib_fallback) and _torch_lib_fallback not in dll_paths:
        dll_paths.insert(0, _torch_lib_fallback)

    existing = os.environ.get("PATH", "")
    new_paths = os.pathsep.join(p for p in dll_paths if os.path.isdir(p))
    if new_paths:
        os.environ["PATH"] = new_paths + os.pathsep + existing

    if hasattr(os, "add_dll_directory"):
        for dll_path in dll_paths:
            if os.path.isdir(dll_path):
                try:
                    os.add_dll_directory(dll_path)
                except Exception:
                    pass

    if os.environ.get("PANOCOUNT_DLL_DEBUG") == "1":
        print("[DLL FIX] PATH prepended with conda DLL dirs")

# Project root must be importable (models.*, data.*, inference_wrappers.*)
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
