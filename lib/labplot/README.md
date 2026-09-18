# labplot

Lab-wide matplotlib plotting style and helpers, shared across projects.

```python
from labplot import my_plot_frame, cbar_postprocess, savefigure
from labplot import coolwarm_white, coolwarm_gray, coolwarm_black

fig, axs = my_plot_frame(60, 45, xlabel="Frequency (GHz)", ylabel="Contrast")
```

Importing `labplot` sets global rcParams (Arial, TrueType PDF/PS fonts type 42,
STIX math).

## Contents

| Name | Purpose |
|---|---|
| `my_plot_frame(xsize, ysize, ..., dpi=300)` | Figure whose axes have an exact size in **mm**; returns `(fig, axs)` with `axs` a 2D array. `dpi` is the on-screen raster density only — fonts are in points, so proportions are `dpi`-independent |
| `cbar_postprocess(cbar)` | Apply lab tick/label style to a colorbar |
| `savefigure(fig, path, form="pdf", dpi=600)` | Save with lab export defaults (600 dpi, tight bbox, transparent) |
| `coolwarm_white` / `coolwarm_gray` / `coolwarm_black` | Diverging blue–*/*–red colormaps |
| `create_custom_cmap`, `create_custom_cmap_v1` | Builders for the above |
| `tqdm_dict` | Consistent `tqdm` bar styling (`dict(ncols=100, colour="#87ceeb")`) |

## Provenance

Extracted from the group's `basic_header_202601.py` (2026-01). That header was a
kitchen-sink import module (FFT via `myFFT`, wavelet synchrosqueezing via
`ssqueezepy`, `corner`, `lmfit` models, THz background functions). Only the
**plotting/style** layer was carried over here, verbatim where kept. Dropped,
because they are analysis code rather than plotting style:

- imports: `pandas`, `scipy.signal`, `lmfit`, `myFFT`, `corner`, `ssqueezepy`
- functions: `nm2ev`, `bg_func`, `find_nearest`, `absmaxND`, `cycle_data`,
  `um2ps`, `viz` (wavelet spectrogram)

If any of those are needed later they should land in a dedicated analysis
package (e.g. `lib/thzfft`, `lib/myfft`), not here.

## Versioning

Tagged releases (`v0.1.0`, ...). Depend on an exact tag from a project once it
is stabilising toward submission.
