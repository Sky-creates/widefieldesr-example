"""labplot — lab-wide matplotlib plotting style and helpers.

Extracted from the group's ``basic_header_202601.py`` and reduced to a pure
plotting/style layer: global rcParams, a millimetre-precise figure/axis frame
(:func:`my_plot_frame`), colorbar styling (:func:`cbar_postprocess`), a figure
saver (:func:`savefigure`), and the ``coolwarm_*`` diverging colormaps.

Numerical / analysis helpers from the original header (FFT, wavelet, corner,
lmfit models, background functions) are intentionally *not* here — they belong
in analysis packages, not the plotting style.

Importing this module sets global ``matplotlib`` rcParams (Arial, TrueType PDF
fonts, STIX math).
"""

from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    "tqdm_dict",
    "my_plot_frame",
    "cbar_postprocess",
    "savefigure",
    "coolwarm_black",
    "coolwarm_gray",
    "coolwarm_white",
    "create_custom_cmap",
    "create_custom_cmap_v1",
]

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
mpl.rcParams["font.family"] = ["Arial"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["mathtext.fontset"] = "stix"

#: Consistent styling for ``tqdm`` progress bars across the group's code.
tqdm_dict = dict(ncols=100, colour="#87ceeb")


# ---------------------------------------------------------------------------
# Figure / axis frame
# ---------------------------------------------------------------------------
def my_plot_frame(
    xsize,
    ysize,
    xlim=None,
    ylim=None,
    xlabel="x",
    ylabel="y",
    xsubplots=1,
    ysubplots=1,
    axial=False,
    margins=None,
    dpi=300,
):
    """Create a figure whose *axes* have an exact physical size.

    ``xsize``/``ysize`` are the axis dimensions in millimetres (per subplot).
    ``margins`` is ``[left, right, top, bottom]`` in millimetres; defaults to
    ``[15, 5, 5, 12]`` (``[5, 5, 5, 5]`` when ``axial``).

    ``dpi`` sets the on-screen raster density (default 300 — comfortable for
    notebook display). Font sizes are in points, so they keep the same physical
    proportion at any ``dpi``; :func:`savefigure` re-rasterises at its own
    (higher) ``dpi`` for publication export.

    Returns ``(fig, axs)`` with ``axs`` always a 2D array (``squeeze=False``).
    """
    mm = 1 / 25.4

    if margins is None:
        margins = [5, 5, 5, 5] if axial else [15, 5, 5, 12]

    left_margin = margins[0] * mm
    right_margin = margins[1] * mm
    top_margin = margins[2] * mm
    bottom_margin = margins[3] * mm

    total_axis_width = xsize * mm * xsubplots
    total_axis_height = ysize * mm * ysubplots

    fig_width = left_margin + total_axis_width + right_margin
    fig_height = bottom_margin + total_axis_height + top_margin

    if axial:
        fig, axs = plt.subplots(
            ysubplots,
            xsubplots,
            subplot_kw={"projection": "polar"},
            figsize=(fig_width, fig_height),
            dpi=dpi,
            squeeze=False,
        )
    else:
        fig, axs = plt.subplots(
            ysubplots,
            xsubplots,
            figsize=(fig_width, fig_height),
            dpi=dpi,
            squeeze=False,
        )

    plt.subplots_adjust(
        left=left_margin / fig_width,
        right=1 - right_margin / fig_width,
        top=1 - top_margin / fig_height,
        bottom=bottom_margin / fig_height,
    )

    for row in axs:
        for ax in row:
            if not axial:
                ax.set_xlabel(xlabel, fontsize=7)
                ax.set_ylabel(ylabel, fontsize=7)
            ax.tick_params(
                axis="both",
                which="major",
                direction="in",
                length=2,
                width=1,
                colors="k",
                grid_color="k",
                grid_alpha=0.5,
                labelsize=7,
            )
            if axial:
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.tick_params(grid_color="k", grid_alpha=0, labelsize=7)
            if xlim:
                ax.set_xlim(xlim[0], xlim[1])
            if ylim:
                ax.set_ylim(ylim[0], ylim[1])

    if not axial:
        for i in range(xsubplots):
            for j in range(ysubplots):
                axs[j, i].ticklabel_format(
                    axis="y", style="sci", scilimits=[-2, 3], useMathText=True
                )
                axs[j, i].yaxis.offsetText.set_fontsize(7)

    return (fig, axs)


def cbar_postprocess(cbar):
    """Apply the lab tick/label style to a colorbar in place."""
    cbar.ax.tick_params(labelsize=7, length=2, width=1, direction="in")
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    cbar.ax.yaxis.offsetText.set_fontsize(7)


def savefigure(fig, directory, form="pdf", dpi=600):
    """Save ``fig`` to ``directory`` with the lab's export defaults.

    ``dpi`` defaults to 600 (publication) regardless of the figure's on-screen
    ``dpi``; ``savefig`` re-rasterises any raster content at this density.
    """
    fig.savefig(
        directory,
        dpi=dpi,
        edgecolor="b",
        orientation="portrait",
        format=form,
        bbox_inches="tight",
        transparent=True,
        pad_inches=0.1,
        metadata=None,
    )


# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------
red = np.array((140 / 255, 47 / 255, 57 / 255)) * 1.4
blue = np.array((2 / 255, 62 / 255, 125 / 255)) * 1.4
black = (0.1, 0.1, 0.1)
gray = (0.8, 0.8, 0.8)
white = (1, 1, 1)


def create_custom_cmap(start_color, mid_color, end_color, name):
    """Three-node linear colormap: ``start`` → ``mid`` → ``end``."""
    colors = [start_color, mid_color, end_color]
    nodes = [0.0, 0.5, 1.0]
    return LinearSegmentedColormap.from_list(name, list(zip(nodes, colors)))


def create_custom_cmap_v1(red, black, blue, name="red_black_blue_improved1"):
    """Diverging colormap with a narrow dark band in the middle."""
    colors = [
        red,
        tuple(0.7 * x for x in red),
        tuple(0.4 * x for x in red),
        black,
        tuple(0.4 * x for x in blue),
        tuple(0.7 * x for x in blue),
        blue,
    ]
    positions = [0, 0.3, 0.45, 0.5, 0.55, 0.7, 1.0]
    return LinearSegmentedColormap.from_list(name, list(zip(positions, colors)))


coolwarm_black = create_custom_cmap_v1(blue, black, red, "red_black_blue")
coolwarm_gray = create_custom_cmap(blue, gray, red, "red_gray_blue")
coolwarm_white = create_custom_cmap(blue, white, red, "red_white_blue")
