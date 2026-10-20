"""Shared chart style for the notebooks.

    from src.plotting import BLUE, ORANGE, SEQ, fmt_k, heat_axes, use_style
    use_style()
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap

# One blue for single-series bars, a fixed categorical order, a one-hue blue ramp for heatmaps
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#a3a29c"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = LinearSegmentedColormap.from_list(
    "blue_seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Axis formatter: 1500 -> 1.5k, 2000000 -> 2M
fmt_k = mtick.FuncFormatter(
    lambda v, _: f"{v/1e6:g}M" if v >= 1e6 else f"{v/1e3:g}k" if v >= 1e3 else f"{v:g}")


def use_style() -> None:
    """Light grid, no top/right spines, bold titles."""
    plt.rcParams.update({
        "figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.edgecolor": "#8a8984", "axes.titleweight": "bold", "axes.titlesize": 12,
    })


def heat_axes(ax, months: bool = True) -> None:
    """Clean up a seaborn heatmap: no grid lines through cells, horizontal row labels.
    With months=True the columns are month numbers (1-12) and get month names. The names
    come from each column's own label, so a missing month can't shift them."""
    ax.grid(False)
    ax.tick_params(axis="y", rotation=0)
    if months:
        ticks = ax.get_xticklabels()
        ax.set_xticklabels([MONTHS[int(float(t.get_text())) - 1] for t in ticks], rotation=0)
