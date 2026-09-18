"""Plotting helpers for widefield ESR maps and spectra (lab style via ``labplot``)."""

from __future__ import annotations

import numpy as np
import matplotlib.patches as mpatches

from labplot import my_plot_frame, cbar_postprocess

__all__ = [
    "plot_image",
    "plot_image_with_roi",
    "plot_dual_spectrum",
    "plot_spectrum_with_fit",
]


def plot_image(img, title="", cmap="hot", vmin=None, vmax=None,
               center_zero=False, crop_roi=True, roi_margin=8):
    """Plot a 2D image with colorbar using the lab style.

    ``center_zero``: symmetrise colorbar around zero (``vmin = -vmax = max|img|``).
    ``crop_roi``: crop display to the bounding box of finite pixels (auto-skips
    full-frame images with no NaN, so safe as default).
    ``roi_margin``: extra pixels of padding around the bounding box.
    """
    fig, axs = my_plot_frame(
        60, 60, xlabel="x (px)", ylabel="y (px)", margins=[14, 16, 8, 14]
    )
    ax = axs[0, 0]

    finite_mask = np.isfinite(img)
    valid = img[finite_mask]

    if center_zero and vmin is None and vmax is None:
        abs_max = float(np.abs(valid).max()) if len(valid) > 0 else 1.0
        vmin, vmax = -abs_max, abs_max

    im = ax.imshow(img, cmap=cmap, origin="upper", aspect="equal", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=7)

    if crop_roi and not finite_mask.all():
        rows, cols = np.where(finite_mask)
        ax.set_xlim(cols.min() - roi_margin, cols.max() + roi_margin)
        ax.set_ylim(rows.max() + roi_margin, rows.min() - roi_margin)  # upper origin

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar_postprocess(cbar)
    return fig


def plot_image_with_roi(img, roi_type=None, roi_params=None, title="", cmap="hot",
                        vmin=None, vmax=None, pixels=None):
    """Plot image with optional cyan ROI overlay and coloured pixel crosshairs.

    ``roi_type``: ``'rect'`` | ``'circle'`` | ``'ellipse'``.
    ``roi_params``: ``(cx, cy, a, b)`` for rect/ellipse (centred); ``(cx, cy, r)`` for circle.
    ``pixels``: list of ``{"name": str, "x": int, "y": int}`` to draw crosshairs.
    """
    fig, axs = my_plot_frame(
        60, 60, xlabel="x (px)", ylabel="y (px)", margins=[14, 16, 8, 14]
    )
    ax = axs[0, 0]
    im = ax.imshow(img, cmap=cmap, origin="upper", aspect="equal", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=7)

    if roi_type == "rect" and roi_params is not None:
        cx, cy, a, b = roi_params
        ax.add_patch(mpatches.Rectangle(
            (cx - a, cy - b), 2 * a, 2 * b, lw=1, edgecolor="cyan", facecolor="none"
        ))
    elif roi_type == "circle" and roi_params is not None:
        cx, cy, r = roi_params
        ax.add_patch(mpatches.Circle(
            (cx, cy), r, lw=1, edgecolor="cyan", facecolor="none"
        ))
    elif roi_type == "ellipse" and roi_params is not None:
        cx, cy, a, b = roi_params
        ax.add_patch(mpatches.Ellipse(
            (cx, cy), 2 * a, 2 * b, lw=1, edgecolor="cyan", facecolor="none"
        ))

    _PX_COLORS = ["#ffffff", "#00ffff", "#00ff00", "#ffff00", "#ff69b4", "#ffa500"]
    for i, px in enumerate(pixels or []):
        c = _PX_COLORS[i % len(_PX_COLORS)]
        ax.axvline(px["x"], color=c, lw=0.8, ls=":")
        ax.axhline(px["y"], color=c, lw=0.8, ls=":")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar_postprocess(cbar)
    return fig


def plot_dual_spectrum(
    freq_GHz,
    top_data,
    bot_data,
    top_label="Fluorescence (counts)",
    bot_label="Contrast dF/F",
    title="",
    freq_marker=None,
    top_color="k",
    bot_color="#2c5f8a",
):
    """Two-panel spectrum: raw fluorescence on top, ESR contrast on bottom."""
    fig, axs = my_plot_frame(
        90, 32, xlabel="Frequency (GHz)", ylabel=top_label, ysubplots=2
    )
    axs[0, 0].plot(freq_GHz, top_data, color=top_color, lw=1)
    axs[0, 0].set_ylabel(top_label, fontsize=7)
    if title:
        axs[0, 0].set_title(title, fontsize=7)
    axs[1, 0].plot(freq_GHz, bot_data, color=bot_color, lw=1)
    axs[1, 0].set_ylabel(bot_label, fontsize=7)
    axs[1, 0].set_xlabel("Frequency (GHz)", fontsize=7)
    if freq_marker is not None:
        for ax in axs[:, 0]:
            ax.axvline(freq_marker, color="r", lw=0.8, ls="--", alpha=0.7)
    return fig


def plot_spectrum_with_fit(freq_GHz, spectrum, fit_result=None, title="",
                           color="#2c5f8a", fit_color="r"):
    """Single-panel contrast spectrum with optional fit curve overlay."""
    fig, axs = my_plot_frame(90, 40, xlabel="Frequency (GHz)", ylabel="Contrast dF/F")
    ax = axs[0, 0]
    ax.plot(freq_GHz, spectrum, color=color, lw=1, label="data")
    if fit_result is not None:
        ax.plot(freq_GHz, fit_result.best_fit, color=fit_color, lw=0.8,
                ls="--", label="fit")
        ax.legend(fontsize=6, frameon=False)
    if title:
        ax.set_title(title, fontsize=7)
    return fig
