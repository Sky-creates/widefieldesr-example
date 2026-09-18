"""Data loading and analysis-sidecar I/O.

``.mat`` files are read from :func:`~widefieldesr.config.data_root` and never
written to. Derived files are split by kind:

- ``metadata.json`` (ROI / pixel / fit-init definitions) → under
  :func:`~widefieldesr.config.analysis_root` (git-tracked provenance)
- ``*_fit.npz`` and plot PDFs → under
  :func:`~widefieldesr.config.output_root` (regenerable, gitignored)

Both mirror the data file's path relative to the data root, in a
``<stem>_meta/`` folder.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import scipy.io as sio

from .config import analysis_root, output_root, relative_to_data

__all__ = [
    "load_esr",
    "load_field_txt",
    "metadata_dir",
    "output_dir",
    "load_metadata",
    "save_metadata",
    "save_fit_maps",
    "load_fit_maps",
    "export_session_plots",
]


# ============================================================
# Data loading
# ============================================================

def load_esr(filepath):
    """Load a widefield ESR ``.mat`` file.

    Returns a dict with ``signal`` (ny, nx, n_freq) plus frequency axes and
    acquisition metadata.
    """
    mat = sio.loadmat(os.fspath(filepath))
    g = mat["gWide"][0, 0]
    signal = g["signal"]           # (ny, nx, n_freq) float64
    freq_hz = g["SweepParam"][0]   # (n_freq,) Hz
    return {
        "signal": signal,
        "freq_hz": freq_hz,
        "freq_GHz": freq_hz / 1e9,
        "N": int(g["N"][0, 0]),
        "frame_rate": int(g["AcquisitionFrameRate"][0, 0]),
        "average": int(g["Average"][0, 0]),
        "bOn": int(g["bOn"][0, 0]),
    }


def _coerce(value: str):
    """Best-effort scalar coercion for a ``key=value`` sidecar field."""
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def load_field_txt(mat_path):
    """Read the ``<stem>.field.txt`` sidecar next to an ESR ``.mat`` cube.

    These ``key=value`` sidecars are written by the rig's ``bcal_run.py`` for
    B-field calibration series (applied current, sweep leg/index, MW params,
    original autosave name, sample metadata). Numeric values are coerced to
    ``int``/``float``. Returns ``None`` when the sidecar is absent.
    """
    p = Path(os.fspath(mat_path))
    sidecar = p.with_suffix(".field.txt")
    if not sidecar.exists():
        return None
    out = {}
    for line in sidecar.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = _coerce(value.strip())
    return out


# ============================================================
# Sidecar directories
# ============================================================

def _meta_subpath(data_path) -> Path:
    p = Path(os.fspath(data_path))
    return relative_to_data(p).parent / f"{p.stem}_meta"


def metadata_dir(data_path) -> Path:
    """Directory holding ``metadata.json`` for ``data_path`` (under analysis root)."""
    return analysis_root() / _meta_subpath(data_path)


def output_dir(data_path) -> Path:
    """Directory holding fit maps / plots for ``data_path`` (under output root)."""
    return output_root() / _meta_subpath(data_path)


# ============================================================
# Metadata I/O
# ============================================================

_DEFAULT_META = {
    "last_roi": {"shape": "Rect", "cx": 256, "cy": 256, "a": 100, "b": 80, "n_peaks": 2},
    "pixels": {},
    "rois": {},
}


def load_metadata(data_path):
    fpath = metadata_dir(data_path) / "metadata.json"
    if fpath.exists():
        with open(fpath) as f:
            return json.load(f)
    return copy.deepcopy(_DEFAULT_META)


def save_metadata(data_path, meta):
    mdir = metadata_dir(data_path)
    mdir.mkdir(parents=True, exist_ok=True)
    with open(mdir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


# ============================================================
# Fit-map I/O
# ============================================================

def save_fit_maps(data_path, name, fit_maps):
    odir = output_dir(data_path)
    odir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(odir / f"{name}_fit.npz", **fit_maps)


def load_fit_maps(data_path, name):
    fpath = output_dir(data_path) / f"{name}_fit.npz"
    if not fpath.exists():
        return None
    data = np.load(fpath)
    return dict(data)


def export_session_plots(data_path, name, figs_dict):
    pdir = output_dir(data_path) / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    for label, fig in figs_dict.items():
        fig.savefig(
            pdir / f"{name}_{label}.pdf",
            dpi=600, format="pdf", bbox_inches="tight",
            transparent=True, pad_inches=0.1,
        )
