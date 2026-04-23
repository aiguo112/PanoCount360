"""Dark matplotlib rcParams for embedded plots."""
from __future__ import annotations

import matplotlib


def apply_scientific_dark_theme() -> None:
    matplotlib.rcParams.update(
        {
            "figure.facecolor": "#0A0E1A",
            "axes.facecolor": "#12172A",
            "axes.edgecolor": "#00D4FF",
            "axes.labelcolor": "#E8EAED",
            "axes.titlecolor": "#00D4FF",
            "text.color": "#E8EAED",
            "xtick.color": "#9AA0A6",
            "ytick.color": "#9AA0A6",
            "grid.color": "#00D4FF26",
            "grid.linestyle": "--",
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Roboto", "DejaVu Sans"],
            "font.size": 10,
        }
    )
