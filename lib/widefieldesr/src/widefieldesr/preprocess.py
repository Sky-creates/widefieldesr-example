"""Per-pixel preprocessing: ESR contrast and simple reductions."""

from __future__ import annotations

__all__ = ["compute_contrast", "dip_depth_map", "spatial_avg_spectrum"]


def compute_contrast(signal, ref_indices=None):
    """Per-pixel ESR contrast: ``dF/F = (F - F_ref) / F_ref``.

    ``ref_indices``: freq indices used for the reference; ``None`` uses the
    mean over all frequencies.
    """
    if ref_indices is None:
        F_ref = signal.mean(axis=2, keepdims=True)
    else:
        F_ref = signal[:, :, ref_indices].mean(axis=2, keepdims=True)
    return (signal - F_ref) / F_ref


def dip_depth_map(contrast):
    """Minimum contrast per pixel — negative values mark the ESR dip."""
    return contrast.min(axis=2)


def spatial_avg_spectrum(signal):
    """Mean spectrum averaged over all pixels."""
    return signal.mean(axis=(0, 1))
