"""
style.py | One visual language for every figure, report and dashboard in this repo.

Colours follow a validated, colour-blind-safe categorical order (never cycled),
thin marks, hairline grids and no chart junk. Import `apply()` once per script.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: figures are written to files, never shown
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots, used in this fixed order (slot 1 first). Identity only.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Status colours are reserved for meaning (pass / warn / fail), never for a series.
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
# Sequential (one hue, light -> dark) and diverging (blue <-> grey <-> red) ramps.
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3b1ae", "#e66767", "#b52e2e"]

INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb", "page": "#f9f9f7"}
REGIME_SHADE = "#f3d9cf"  # soft wash for "turbulent regime" bands


def apply() -> None:
    """Set matplotlib defaults to the house style."""
    plt.rcParams.update({
        "figure.figsize": (9, 4.2),
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 9.5,
        "axes.labelcolor": INK["secondary"],
        "axes.edgecolor": INK["axis"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "grid.color": INK["grid"],
        "grid.linewidth": 0.6,
        "xtick.color": INK["muted"],
        "ytick.color": INK["muted"],
        "xtick.labelcolor": INK["secondary"],
        "ytick.labelcolor": INK["secondary"],
        "lines.linewidth": 1.4,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
    })


def save(fig, path) -> str:
    """Save a figure and close it; returns the path as a string."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def subtitle(ax, text: str) -> None:
    """Grey one-line subtitle under the axes title."""
    ax.text(0, 1.01, text, transform=ax.transAxes, fontsize=8.5, color=INK["secondary"], va="bottom")
