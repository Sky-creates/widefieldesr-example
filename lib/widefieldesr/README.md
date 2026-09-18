# widefieldesr

Widefield NV-ESR magnetometry imaging analysis, shared across projects.

```python
import widefieldesr as wh

d = wh.load_esr(wh.data_root() / "Bi-2212-S1/RT/WidefieldESR_2026-8-28-19-10.mat")
contrast = wh.compute_contrast(d["signal"])
mask = wh.make_circle_roi(cx=256, cy=320, radius=80)
maps = wh.fit_roi(contrast, mask, d["freq_GHz"], n_peaks=2, use_gpu="auto")
wh.plot_image(maps["f0_2"] - maps["f0_1"], title="Splitting (GHz)",
              cmap=wh.coolwarm_white, center_zero=True)

# splitting (MHz) -> total field along [100] (G), via the exact NV Hamiltonian
df_to_B100 = wh.make_B100_inverter(D=2877.3, E=3.3)
B_map = df_to_B100((maps["f0_2"] - maps["f0_1"]) * 1e3)
```

## Modules

| Module | Contents |
|---|---|
| `config` | `data_root()`, `analysis_root()`, `output_root()` from env / `.env` |
| `io` | `load_esr`, `load_field_txt` (`.field.txt` current/param sidecar), `metadata_dir`/`output_dir`, `load/save_metadata`, `load/save_fit_maps`, `export_session_plots` |
| `preprocess` | `compute_contrast`, `dip_depth_map`, `spatial_avg_spectrum` |
| `binning` | `bin_mask`, `bin_contrast`, `upsample_maps` (non-overlapping block binning, `bin_mode="block"`) and `smooth_contrast` (mask-aware sliding-window average, `bin_mode="sliding"`) — see `fit_roi` |
| `roi` | `make_{rect,circle,ellipse}_roi`, `roi_spectra`, `roi_avg_spectrum` |
| `fitting` | `fit_esr_spectrum` (lmfit), `fit_roi_cpu` (scipy pool), `fit_roi_gpu` (batched CUDA Adam), `fit_roi` (dispatch), `init_from_spectrum(..., smooth_n=1)`, `eval_model` — N-Lorentzian dips on `offset + slope·f`; `linear_bg=False` (default) pins `slope=0`. `fit_roi(..., x_bin=1, y_bin=1, bin_mode="block")` bins the ROI before fitting: `"block"` (default) — non-overlapping blocks, one fit per block, replicated back onto every raw pixel it covers (fewer/faster fits, blocky output); `"sliding"` — moving-window average centred on every pixel (no speedup, but full resolution, no blockiness). `1, 1` (default) is an exact no-op either way. `fit_roi(..., per_pixel_init=False, init_smooth_n=1)` — `per_pixel_init=True` seeds each pixel/block from its own spectrum's deepest minima instead of one shared init (ignores `init_params`); `init_smooth_n>1` smooths a copy of the spectrum before that peak search only (not the data being fit) — recommended together, since per-pixel auto-init on a noisy individual spectrum can otherwise lock onto a spurious minimum |
| `nv_physics` | Exact S=1 NV Hamiltonian: `nv_lines` / `splitting_of_B100` / `make_B100_inverter` ([100]-cut splitting ↔ field), `nv_hamiltonian`, Barson `stress_to_shifts` + `effective_DE` / `stress_from_DE` / `pressure_from_D_shift` / `sigma_from_D` (D map → equiv. [100] stress), thermal `D_of_T` / `dDdT_of_T`, `stark_shifts`. Energies MHz, field G, stress GPa |
| `plotting` | `plot_image`, `plot_image_with_roi`, `plot_dual_spectrum`, `plot_spectrum_with_fit` |
| `current_reconstruction` | `reconstruct_current(Bz, pixel_size_um, standoff_um, psf=None, kx_max=1.0, ky_max=1.0)` — 2D Fourier/Biot-Savart inversion recovering sheet current `(jx, jy)` (A/m) from a measured `Bz` map (Tesla) at a known sensor standoff. Model-agnostic (any in-plane 2D current, not NV- or superconductor-specific); adapted from Roth, Sepulveda & Wikswo, *J. Appl. Phys.* 65, 361 (1989) / [loganbvh/current-reconstruction](https://github.com/loganbvh/current-reconstruction) (MIT) |

The package namespace also re-exports `plt` and the `labplot` style names
(`my_plot_frame`, `cbar_postprocess`, `coolwarm_white/gray/black`) for notebook
convenience.

## Configuration

Reads three environment variables (via `python-dotenv`, so a project `.env`
works):

| Env var | Meaning | Fallback |
|---|---|---|
| `WIDEFIELDESR_DATA_ROOT` | read-only `.mat` data tree | `~/dropbox/Data/WideFieldESR` |
| `WIDEFIELDESR_ANALYSIS_ROOT` | git-tracked `metadata.json` | `<cwd>/analysis` |
| `WIDEFIELDESR_OUTPUT_ROOT` | regenerable `*_fit.npz` + plot PDFs | `<cwd>/outputs` |

Sidecars mirror each `.mat` file's path relative to the data root, in a
`<stem>_meta/` folder. Data files are never written to.

## GPU

`fit_roi_gpu` runs on CUDA when available (this workstation: RTX 5070 Ti,
Blackwell / sm_120). `torch` is pinned to the CUDA 12.8 wheel index in
`pyproject.toml` (`>=2.7`, required for sm_120). No system CUDA toolkit needed —
the wheel bundles its runtime. On a machine without an NVIDIA GPU it falls back
to CPU.

## Smoke test

```
uv run python -m widefieldesr.smoke               # default acquisition
uv run python -m widefieldesr.smoke /path/to.mat  # explicit file
```

## Versioning

Tagged releases (`v0.1.0`, ...). Depends on `labplot` (editable path source
while co-developing; pin to a tag when a downstream project stabilises).
