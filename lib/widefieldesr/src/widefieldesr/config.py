"""Path configuration, driven by environment variables (loaded from ``.env``).

Three roots, each overridable via an env var:

===========================  ==================================  ==========================
Env var                      Meaning                             Fallback
===========================  ==================================  ==========================
``WIDEFIELDESR_DATA_ROOT``    read-only experimental data tree     ``~/dropbox/Data/WideFieldESR``
``WIDEFIELDESR_ANALYSIS_ROOT`` git-tracked ``metadata.json``       ``<cwd>/analysis``
``WIDEFIELDESR_OUTPUT_ROOT``  regenerable fit maps + plot PDFs     ``<cwd>/outputs``
===========================  ==================================  ==========================

A project sets all three explicitly in its ``.env``; the fallbacks only apply
to ad-hoc use from a bare interpreter.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

_DOTENV_PATH = find_dotenv(usecwd=True)
load_dotenv(_DOTENV_PATH)

#: Directory a relative root is resolved against (the .env's dir, else cwd).
_BASE_DIR = Path(_DOTENV_PATH).parent if _DOTENV_PATH else Path.cwd()

_DEFAULT_DATA_ROOT = "~/dropbox/Data/WideFieldESR"


def _root(env_var: str, default) -> Path:
    p = Path(os.environ.get(env_var) or default).expanduser()
    return p if p.is_absolute() else (_BASE_DIR / p).resolve()


def data_root() -> Path:
    """Root of the read-only experimental data tree."""
    return _root("WIDEFIELDESR_DATA_ROOT", _DEFAULT_DATA_ROOT)


def analysis_root() -> Path:
    """Root for git-tracked analysis provenance (``metadata.json``)."""
    return _root("WIDEFIELDESR_ANALYSIS_ROOT", "analysis")


def output_root() -> Path:
    """Root for regenerable outputs (fit-map ``.npz``, plot PDFs)."""
    return _root("WIDEFIELDESR_OUTPUT_ROOT", "outputs")


def relative_to_data(data_path) -> Path:
    """``data_path`` expressed relative to :func:`data_root`.

    Falls back to just the file name when ``data_path`` lies outside the data
    root, so absolute paths from anywhere still map to a stable sidecar
    location.
    """
    p = Path(os.fspath(data_path)).expanduser()
    try:
        p = p.resolve()
    except OSError:
        pass
    try:
        return p.relative_to(data_root().resolve())
    except (ValueError, OSError):
        return Path(p.name)
