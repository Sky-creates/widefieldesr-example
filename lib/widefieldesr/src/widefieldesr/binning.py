"""Spatial binning for per-pixel ROI fitting.

Two ways to trade spatial resolution for SNR before fitting a ROI, both
selected via :func:`widefieldesr.fitting.fit_roi`'s ``bin_mode``:

- ``"block"`` (default) — :func:`bin_mask` / :func:`bin_contrast` tile the ROI
  into non-overlapping ``x_bin``×``y_bin`` blocks, fit each block once, then
  :func:`upsample_maps` replicates the result back onto the native pixel grid.
  Fewer, better-SNR spectra to fit (faster); the output looks "blocky" —
  piecewise-constant within each block.
- ``"sliding"`` — :func:`smooth_contrast` computes a ``x_win``×``y_win``
  moving-window average centred on *every* pixel; the result is already full
  resolution, so every pixel is still fit individually (no speedup) but from
  a less noisy spectrum, with no blockiness.

Either way the output map has the exact same shape and NaN-outside-mask
convention as an unbinned fit — this is what lets ``fit_roi`` accept
``x_bin=y_bin=1`` (the default) as a no-op regardless of ``bin_mode``, and any
other consumer (the viewer's pixel inspector, its linecut tool, ``save_fit_maps``
/ ``load_fit_maps``) keep working unchanged.

``bin_mask``/``bin_contrast``/``upsample_maps`` generalise (separate x/y
factors, 3D contrast cubes, mask-aware averaging) the ad hoc square-only
``block_mean`` duplicated for *display-time* map smoothing in the
``bi2212-widefield-esr`` project's ``spot2_bfield_summary.py`` /
``spot2_tdep_summary.py`` notebooks.
"""

from __future__ import annotations

import numpy as np

__all__ = ["bin_mask", "bin_contrast", "upsample_maps", "smooth_contrast"]


def _trimmed(ny, nx, y_bin, x_bin):
    """Largest (ny, nx) that divide evenly by (y_bin, x_bin); trims the remainder."""
    return ny - ny % y_bin, nx - nx % x_bin


def bin_mask(mask, y_bin, x_bin):
    """Coarsen a boolean ``(ny, nx)`` mask to ``(ny//y_bin, nx//x_bin)``.

    A block is ``True`` if *any* of its raw pixels are ``True`` — inclusive at
    ROI edges, so a partially-covered boundary block still gets fit (from
    whatever in-mask pixels it has; see :func:`bin_contrast`). Trims any
    remainder row/col that doesn't divide evenly, same convention as the
    project's ad hoc ``block_mean`` helper.
    """
    ny, nx = mask.shape
    ny2, nx2 = _trimmed(ny, nx, y_bin, x_bin)
    blocks = mask[:ny2, :nx2].reshape(ny2 // y_bin, y_bin, nx2 // x_bin, x_bin)
    return blocks.any(axis=(1, 3))


def bin_contrast(contrast, mask, y_bin, x_bin):
    """Block-mean a ``(ny, nx, nf)`` contrast cube to ``(ny//y_bin, nx//x_bin, nf)``.

    Only pixels where ``mask`` is ``True`` contribute to each block's average
    (masked-out pixels are excluded via NaN + ``np.nanmean``, not blindly
    averaged in) — a block with zero in-mask pixels comes out all-NaN for
    every frequency. ``x_bin=y_bin=1`` returns ``contrast`` with out-of-mask
    pixels set to NaN, matching the trimmed shape (a no-op when ``mask`` is
    all ``True``).
    """
    import warnings

    ny, nx, nf = contrast.shape
    ny2, nx2 = _trimmed(ny, nx, y_bin, x_bin)
    c = contrast[:ny2, :nx2, :].astype(float, copy=True)
    c[~mask[:ny2, :nx2]] = np.nan
    blocks = c.reshape(ny2 // y_bin, y_bin, nx2 // x_bin, x_bin, nf)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # blocks entirely outside `mask` are legitimately all-NaN (e.g. every
        # block outside the ROI's bounding region) -- expected, not a bug.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(blocks, axis=(1, 3))


def smooth_contrast(contrast, mask, y_win, x_win):
    """Mask-aware sliding-window average of a ``(ny, nx, nf)`` contrast cube.

    Unlike :func:`bin_contrast` (non-overlapping blocks, one fit per block),
    this computes a ``y_win``×``x_win`` moving-window average **centred on
    every pixel** — the returned cube is already full ``(ny, nx, nf)``
    resolution, so fitting it per pixel (``fit_roi(..., bin_mode="sliding")``)
    gives a smooth map with no blockiness, at the cost of fitting the same
    number of spectra as an unbinned fit (no speedup — every pixel's spectrum
    is still fit individually, just less noisy).

    Only in-mask pixels contribute to each window's average (both the pixel
    values and the window's normalisation are zeroed outside ``mask``, so a
    window straddling the ROI boundary averages only the genuine ROI pixels
    it contains — the frame edge is treated the same way, via zero-padding,
    rather than replicating edge pixels). ``y_win=x_win=1`` is an exact no-op.
    """
    from scipy.ndimage import uniform_filter

    if y_win <= 1 and x_win <= 1:
        return contrast.astype(float, copy=True)

    m = mask.astype(float)
    weighted = contrast.astype(float) * mask[..., None]
    num = uniform_filter(weighted, size=(y_win, x_win, 1), mode="constant", cval=0.0)
    den = uniform_filter(m, size=(y_win, x_win), mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den[..., None]


def upsample_maps(maps, y_bin, x_bin, shape, mask):
    """Replicate coarse per-block maps back onto the native ``shape`` grid.

    ``maps``: dict of coarse 2D arrays (e.g. from ``_build_param_maps`` on the
    binned contrast/mask). Each array is block-replicated ``y_bin``/``x_bin``
    times along its axes (exact integer repetition — not an interpolating
    resize), padded with NaN out to ``shape`` if binning didn't divide it
    evenly, then re-masked with the ORIGINAL fine-resolution ``mask`` so
    genuinely out-of-ROI raw pixels stay NaN even where :func:`bin_mask`'s
    inclusive rule pulled a boundary block into the fit.
    """
    ny, nx = shape
    out = {}
    for key, arr in maps.items():
        rep = np.repeat(np.repeat(arr, y_bin, axis=0), x_bin, axis=1)
        full = np.full((ny, nx), np.nan, dtype=float)
        yr, xr = min(rep.shape[0], ny), min(rep.shape[1], nx)
        full[:yr, :xr] = rep[:yr, :xr]
        full[~mask] = np.nan
        out[key] = full
    return out
