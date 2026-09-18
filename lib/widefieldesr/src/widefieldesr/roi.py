"""Region-of-interest masks and spectrum extraction."""

from __future__ import annotations

import numpy as np

__all__ = [
    "make_rect_roi",
    "make_circle_roi",
    "make_ellipse_roi",
    "roi_spectra",
    "roi_avg_spectrum",
]


def make_rect_roi(x0, y0, w, h, img_shape=(512, 512)):
    """Boolean mask for a rectangular ROI. ``(x0, y0)`` is the top-left corner."""
    mask = np.zeros(img_shape, dtype=bool)
    y1 = min(int(y0) + int(h), img_shape[0])
    x1 = min(int(x0) + int(w), img_shape[1])
    mask[int(y0):y1, int(x0):x1] = True
    return mask


def make_circle_roi(cx, cy, radius, img_shape=(512, 512)):
    """Boolean mask for a circular ROI centred at ``(cx, cy)``."""
    ny, nx = img_shape
    yy, xx = np.ogrid[:ny, :nx]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def make_ellipse_roi(cx, cy, a, b, img_shape=(512, 512)):
    """Boolean mask for an elliptical ROI centred at ``(cx, cy)`` with semiaxes ``a``, ``b``."""
    ny, nx = img_shape
    yy, xx = np.ogrid[:ny, :nx]
    return ((xx - cx) / max(a, 1)) ** 2 + ((yy - cy) / max(b, 1)) ** 2 <= 1


def roi_spectra(data, mask):
    """All pixel spectra within ``mask``: ``(ny, nx, nf)`` → ``(N_pix, nf)``."""
    return data[mask]


def roi_avg_spectrum(data, mask):
    """Mean spectrum over pixels in ``mask``: ``(ny, nx, nf)`` → ``(nf,)``."""
    return data[mask].mean(axis=0)
