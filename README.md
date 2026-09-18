# widefieldesr-example

A minimal, self-contained example of the `widefieldesr` widefield-ODMR
analysis pipeline, from the Yao Lab at Harvard: one marimo notebook, its two
dependencies vendored in (`widefieldesr`, `labplot`), and a small synthetic
sample dataset — so it runs standalone with no access to the original lab
data or Dropbox.

## Why marimo

The notebook is a [marimo](https://marimo.io) notebook, not Jupyter. A few
reasons that matters here:

- **Reactive** — cells re-run automatically when an upstream cell or UI
  widget changes, so the ROI/pixel/fit state shown is always consistent with
  the current inputs (no stale-cell bugs from running cells out of order).
- **Pure Python file** — it's a plain `.py` file (no JSON/ipynb wrapper), so
  it diffs and reviews cleanly in git, and is friendly to coding agents like
  Claude Code — they can just `Read`/`Edit` it like any other source file
  instead of fighting a notebook's JSON cell/output structure.
- **Watch mode** — `uv run marimo edit --watch
  notebooks/widefield_esr_analysis.py` reloads the notebook from disk and
  re-runs affected cells whenever the file changes on disk, so an editor or
  agent editing the `.py` file directly shows up live in the running
  notebook.
- **Built-in UI widgets** — the ROI/Pixel/Fitting Managers in this notebook
  are just `mo.ui.*` elements (sliders, tables, buttons); no separate
  ipywidgets setup needed.

## What's here

```
data/sample_odmr.mat        synthetic ODMR cube (512x512 px, 21 freq points,
                             2.80-2.95 GHz) -- shaped like a real gWide file
lib/widefieldesr/            vendored copy of the analysis library
lib/labplot/                 vendored copy of the plotting-style library
notebooks/widefield_esr_analysis.py   the notebook itself
```

The sample is synthetic (a smoothly-varying Lorentzian dip + noise per
pixel), not real data — it exists so the notebook has something to load
and every UI default (ROI center/size, pixel picker, etc.) lines up
sensibly out of the box.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run marimo edit notebooks/widefield_esr_analysis.py
```

That's it — no `.env` needed. The notebook defaults to
`data/sample_odmr.mat`; the file browser at the top lets you point at a
different `.mat` file (same `gWide` struct layout) if you want to load
your own data instead.

## What the notebook does

1. **Load** an ODMR cube (`signal[y, x, freq]`) via `widefieldesr.load_esr`.
2. **ROI Manager** — define a rectangle/circle/ellipse ROI, average the
   spectrum over it.
3. **Pixel Manager** — pick individual pixels, view their raw spectra.
4. **Fitting Manager** — fit a sum of Lorentzian dips (initial guesses:
   center `f0`, amplitude `A`, linewidth `gamma`) to the ROI-averaged
   spectrum, then run the same fit per-pixel to build 2D maps of dip
   depth / center frequency / linewidth.

Everything (ROI/pixel/fit-init definitions) is saved to a small
`analysis/metadata.json` next to the data, so re-opening the notebook
picks up where you left off; per-pixel fit result arrays go to
`outputs/` — both directories are gitignored/regenerable.

## Suggested next steps

This notebook is scoped to **one dataset at a time** — load a `.mat`, define
ROIs/pixels, fit, done. Once you've run it over several datasets (several
temperatures, several samples, before/after some change), each one leaves
behind its own `analysis/<name>_meta/metadata.json` (saved ROIs/pixels/fit
inits) and `outputs/<name>_meta/<name>_fit.npz` (the 2D fit-parameter maps,
via `widefieldesr.load_fit_maps`). The natural next step is a **second
notebook** that loads several of these `*_fit.npz` files together and
compares them — e.g. dip depth/center-frequency maps side by side, or a
summary quantity (mean splitting, a linecut, a derived pressure/field value)
plotted across temperature or sample. Keep that as its own notebook rather
than growing this one, so each stays focused: one dataset in, one comparison
across datasets out.

## Notes for reuse

- `widefieldesr`/`labplot` are vendored (copied, not a git submodule) so
  this folder works standing alone. If you already have your own
  research workspace with these as shared `lib/` packages, you may
  prefer to point `pyproject.toml`'s `[tool.uv.sources]` at those instead
  of the vendored copies here.
- Two small portability fixes were made to the notebook copy versus the
  original lab version: the frequency-axis length was read from the
  wrong metadata field (`N`, the averaging count, rather than
  `len(freq_GHz)`), and the ROI-mask builders need `img_shape` passed
  explicitly rather than relying on a library-level 512x512 default —
  both are only visible on datasets whose shape/size differs from what
  the original lab data happened to always have.
