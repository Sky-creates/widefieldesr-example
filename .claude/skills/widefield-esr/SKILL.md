---
name: widefield-esr
description: Working with the widefieldesr widefield-ODMR analysis notebook and library in this example — loading ODMR .mat cubes, defining ROIs/pixels, fitting Lorentzian dips, building 2D fit maps, and NV physics conversions (B-field, pressure). Use when the user asks to analyze ODMR/NV-magnetometry data, extend this notebook, or debug widefieldesr/labplot code in this repo.
---

# widefieldesr widefield-ODMR analysis

This repo is a minimal, standalone bundle: `notebooks/widefield_esr_analysis.py`
(a marimo notebook) + vendored `lib/widefieldesr` and `lib/labplot` + a small
synthetic sample at `data/sample_odmr.mat`. Everything runs with `uv run`, no
external data access required.

## Running it

```bash
uv sync
uv run marimo edit notebooks/widefield_esr_analysis.py   # interactive
uv run python notebooks/widefield_esr_analysis.py         # headless smoke test
```

The headless run is the fast way to check nothing is broken after an edit —
it executes every cell top to bottom and exits nonzero on any exception.

## Data shape and format

`widefieldesr.load_esr(path)` reads a MATLAB `gWide` struct
(`scipy.io.loadmat`) and returns a dict with:
- `signal`: `(ny, nx, n_freq)` float array, one ODMR spectrum per pixel.
- `freq_GHz`: `(n_freq,)` frequency axis in GHz.
- other acquisition metadata (averaging count `N`, frame rate, etc.).

`n_freq` must come from `len(freq_GHz)`, not `_data["N"]` — `N` is the
averaging count and is unrelated to the frequency axis length. (This was a
real bug found while building this example; already fixed in this copy.)

## ROI and pixel-mask gotcha

`widefieldesr.make_rect_roi` / `make_circle_roi` / `make_ellipse_roi` default
to `img_shape=(512, 512)` — the original lab camera's FOV. Any code building
a mask for a differently-shaped image (like the small synthetic sample, if
you shrink it) **must** pass `img_shape=signal.shape[:2]` explicitly, or
`roi_avg_spectrum`/`fit_roi*` will raise
`IndexError: boolean index did not match indexed array`. The current sample
is deliberately kept at 512x512 so the notebook's ROI/pixel UI defaults
(center, radius, etc.) are meaningful out of the box; if you regenerate the
sample at a different resolution, also check the ROI number-input defaults
near the ROI Manager section of the notebook.

## Core library modules (`lib/widefieldesr/src/widefieldesr/`)

- `io.py` — `load_esr`, `load_metadata`/`save_metadata` (per-dataset
  `analysis/metadata.json`: saved ROIs, pixels, fit-init guesses),
  `save_fit_maps`/`load_fit_maps` (per-dataset `.npz` fit results),
  `export_session_plots`.
- `preprocess.py` — `compute_contrast`, `spatial_avg_spectrum`,
  `dip_depth_map`.
- `roi.py` — ROI mask builders + `roi_avg_spectrum`.
- `fitting.py` — `fit_esr_spectrum` (single spectrum, sum of `n_peaks`
  Lorentzian dips + optional linear background), `fit_roi_cpu`/`fit_roi_gpu`/
  `fit_roi` (per-pixel fitting across an ROI → 2D parameter maps), `_auto_init`
  for initial-guess seeding.
- `nv_physics.py` — the NV Hamiltonian (`nv_hamiltonian`, `nv_lines`,
  `splitting_of_B100`) for converting ODMR line positions to magnetic field,
  and stress/pressure conversions (`sigma_from_D`, `hydrostatic_D_shift`,
  `D_of_T`) for converting the zero-field-splitting shift to pressure or
  temperature.
- `binning.py` — spatial binning/smoothing/upsampling of fit-map arrays.
- `plotting.py` — thin plotting helpers on top of `labplot`.

## Notebook structure

The notebook has three linked UI sections, each backed by `analysis/metadata.json`:
1. **ROI Manager** — define/save a rectangle, circle, or ellipse ROI; average
   spectrum over it.
2. **Pixel Manager** — pick and save individual pixels; view raw per-pixel
   spectra.
3. **Fitting Manager** — fit the ROI-averaged spectrum (to get initial
   guesses), then fit every pixel in the ROI to build 2D maps of dip
   depth/center/linewidth.

When extending the notebook, follow the existing pattern: one cell defines a
`mo.ui.*` widget, a downstream cell reacts to `.value`, and anything meant to
persist goes through `save_metadata`/`save_fit_maps` rather than a bare
global. Marimo cells are dependency-ordered automatically from variable
names, not top-to-bottom position.

## Common tasks

- **Add a new derived quantity to a fit map** (e.g. pressure from center
  frequency): compute it in a new cell downstream of the fit-map cell, using
  `nv_physics.sigma_from_D`/`hydrostatic_D_shift`; don't refit.
- **Test against non-512x512 data**: regenerate `data/sample_odmr.mat` at the
  new shape, then check both the ROI mask builders (pass `img_shape`
  explicitly) and the ROI Manager's default slider values before assuming it
  works — both are hardcoded default-512 pitfalls.
- **Verify a change didn't break the notebook**: `uv run marimo check
  notebooks/widefield_esr_analysis.py` (static check) and `uv run python
  notebooks/widefield_esr_analysis.py` (headless execution) — both should
  exit clean/0.
