"""N-Lorentzian dip fitting of NV ESR spectra.

Model::

    contrast(f) = offset + slope·f - Σᵢ  Aᵢ (γᵢ/2)² / [(f - f₀ᵢ)² + (γᵢ/2)²]

The ``slope·f`` linear background is opt-in: every fitting entry point takes
``linear_bg=False`` by default, which pins ``slope = 0`` and reproduces the
plain constant-offset fit exactly. Pass ``linear_bg=True`` to let it vary.
``eval_model`` reads ``slope`` from ``init_params`` (defaulting to 0), so fit
records / metadata written before this parameter existed load unchanged.

Three entry points:

- :func:`fit_esr_spectrum` — one spectrum, ``lmfit`` least-squares
- :func:`fit_roi_cpu` — per-pixel over a ROI, ``scipy`` + process pool
- :func:`fit_roi_gpu` — per-pixel over a ROI, one batched PyTorch Adam pass on CUDA

:func:`fit_roi` dispatches between the last two.
"""

from __future__ import annotations

import numpy as np

from labplot import tqdm_dict

__all__ = [
    "init_from_spectrum",
    "eval_model",
    "fit_esr_spectrum",
    "fit_roi_cpu",
    "fit_roi_gpu",
    "fit_roi",
]


# ============================================================
# Internal helpers (module-level for pickling)
# ============================================================

def _lorentzian_dip_curve_fit(freq, *params):
    """Flat-param Lorentzian dip sum for ``scipy.optimize.curve_fit``.

    ``params = (offset, [slope], A1, f01, gamma1, A2, f02, gamma2, ...)`` — the
    optional ``slope`` element is present iff ``(len(params) - 1) % 3 == 1``.
    """
    if (len(params) - 1) % 3 == 1:
        offset, slope, dip = float(params[0]), float(params[1]), params[2:]
    else:
        offset, slope, dip = float(params[0]), 0.0, params[1:]
    n_peaks = len(dip) // 3
    result = offset + slope * freq
    for i in range(n_peaks):
        A = float(dip[3 * i])
        f0 = float(dip[3 * i + 1])
        gamma = float(dip[3 * i + 2])
        result = result - A * (gamma / 2) ** 2 / ((freq - f0) ** 2 + (gamma / 2) ** 2)
    return result


def _fit_pixel_worker(args):
    """Per-pixel scipy fitting worker. Module-level for multiprocessing pickling."""
    spectrum, freq_GHz, p0, bounds = args
    from scipy.optimize import curve_fit
    try:
        popt, _ = curve_fit(
            _lorentzian_dip_curve_fit, freq_GHz, spectrum,
            p0=p0, bounds=bounds, maxfev=3000,
        )
        return popt
    except Exception:
        return np.array(p0)


def _auto_init(freq_GHz, spectrum, n_peaks, smooth_n=1):
    """Estimate initial dip parameters from the deepest minima in ``spectrum``.

    ``smooth_n`` (default 1, no-op): before locating minima (and reading off
    their depth/offset), apply an ``smooth_n``-point moving average to a
    *copy* of ``spectrum`` — purely to make the peak search more robust
    against noise on an individual, low-SNR spectrum (e.g. one raw pixel);
    the spectrum actually being fit is never touched by this.
    """
    from scipy.signal import find_peaks

    loc = spectrum if smooth_n <= 1 else _smooth_1d(spectrum, smooth_n)
    neg = -(loc - loc.max())  # dips become positive peaks
    min_dist = max(2, len(freq_GHz) // max(1, n_peaks * 4))
    peaks, _ = find_peaks(neg, height=0.05 * neg.max(), distance=min_dist)
    if len(peaks) >= n_peaks:
        top = np.argsort(-neg[peaks])[:n_peaks]
        peak_locs = sorted(peaks[top].tolist())
    else:
        peak_locs = np.linspace(
            len(freq_GHz) // 5, 4 * len(freq_GHz) // 5, n_peaks, dtype=int
        ).tolist()
    init = {"offset": float(loc.mean()), "slope": 0.0}
    for j, idx in enumerate(peak_locs, 1):
        depth = max(abs(float(loc[idx] - loc.mean())), 1e-5)
        init[f"A{j}"] = depth
        init[f"f0{j}"] = float(freq_GHz[idx])
        init[f"gamma{j}"] = 0.01  # 10 MHz default linewidth
    return init


def _smooth_1d(spectrum, n):
    """``n``-point centred moving average (edge-safe), for peak-location only."""
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(spectrum, size=int(n), mode="nearest")


def _p0_vector(p0_dict, n_peaks, linear_bg):
    """Flatten an init-params dict into the ``(offset, [slope], A, f0, gamma, ...)``
    list :func:`_fit_pixel_worker` / the GPU init tensor expect."""
    p0 = [p0_dict["offset"]]
    if linear_bg:
        p0.append(float(p0_dict.get("slope", 0.0)))
    for i in range(1, n_peaks + 1):
        p0 += [p0_dict[f"A{i}"], p0_dict[f"f0{i}"], p0_dict[f"gamma{i}"]]
    return p0


def _param_bounds(n_peaks, linear_bg, freq_GHz):
    """Fixed ``(lo, hi)`` bounds — same for every pixel regardless of init source."""
    lo, hi = [-0.5], [0.5]
    if linear_bg:
        lo.append(-np.inf)
        hi.append(np.inf)
    for _ in range(n_peaks):
        lo += [0, float(freq_GHz[0]), 1e-4]
        hi += [1.0, float(freq_GHz[-1]), 0.5]
    return lo, hi


def _build_param_maps(params_np, mask, img_shape, n_peaks, freq_GHz, spectra, linear_bg=False):
    """Pack per-pixel fit arrays into 2D maps (NaN outside mask).

    ``params_np`` column layout: ``offset``, then ``slope`` iff ``linear_bg``,
    then ``(A, f0, gamma)`` per peak. A ``slope`` map is always returned (all
    zeros when ``linear_bg`` is False).
    """
    result = {}
    offset_map = np.full(img_shape, np.nan)
    offset_map[mask] = params_np[:, 0]
    result["offset"] = offset_map

    slope_map = np.full(img_shape, np.nan)
    if linear_bg:
        slope_map[mask] = params_np[:, 1]
        dip0 = 2
    else:
        slope_map[mask] = 0.0
        dip0 = 1
    result["slope"] = slope_map

    for i in range(1, n_peaks + 1):
        base = dip0 + 3 * (i - 1)
        A_map = np.full(img_shape, np.nan)
        f0_map = np.full(img_shape, np.nan)
        g_map = np.full(img_shape, np.nan)
        A_map[mask] = params_np[:, base]
        f0_map[mask] = params_np[:, base + 1]
        g_map[mask] = params_np[:, base + 2]
        result[f"A_{i}"] = A_map
        result[f"f0_{i}"] = f0_map
        result[f"gamma_{i}"] = g_map

    # Reduced chi-squared
    pred = params_np[:, 0:1] + np.zeros_like(spectra)
    if linear_bg:
        pred = pred + params_np[:, 1:2] * freq_GHz
    for i in range(n_peaks):
        base = dip0 + 3 * i
        A = params_np[:, base : base + 1]
        f0 = params_np[:, base + 1 : base + 2]
        gamma = params_np[:, base + 2 : base + 3]
        pred = pred - A * (gamma / 2) ** 2 / (
            (freq_GHz - f0) ** 2 + (gamma / 2) ** 2
        )
    dof = max(1, spectra.shape[1] - dip0 - 3 * n_peaks)
    redchi_map = np.full(img_shape, np.nan)
    redchi_map[mask] = ((spectra - pred) ** 2).sum(axis=1) / dof
    result["redchi"] = redchi_map
    return result


# ============================================================
# Public API — single spectrum
# ============================================================

def init_from_spectrum(freq_GHz, spectrum, n_peaks=2, smooth_n=1):
    """Auto-estimate Lorentzian dip init params from the deepest minima.

    ``smooth_n`` (default 1, no-op): smooth a copy of ``spectrum`` before
    locating minima — see :func:`fit_roi`'s ``init_smooth_n``.
    Returns ``{offset, slope, A1, f01, gamma1, A2, f02, gamma2, ...}`` (``slope``
    seeded at 0).
    """
    return _auto_init(freq_GHz, spectrum, n_peaks, smooth_n=smooth_n)


def eval_model(freq_GHz, init_params, n_peaks):
    """Evaluate the N-Lorentzian dip model at given parameters (no fitting).

    ``slope`` is optional in ``init_params`` and defaults to 0, so records
    written before the linear background existed evaluate identically.
    """
    result = np.full_like(freq_GHz, init_params["offset"], dtype=float)
    result = result + init_params.get("slope", 0.0) * freq_GHz
    for i in range(1, n_peaks + 1):
        A = init_params[f"A{i}"]
        f0 = init_params[f"f0{i}"]
        gamma = init_params[f"gamma{i}"]
        result -= A * (gamma / 2) ** 2 / ((freq_GHz - f0) ** 2 + (gamma / 2) ** 2)
    return result


def fit_esr_spectrum(freq_GHz, spectrum, n_peaks=2, init_params=None, linear_bg=False):
    """Fit the N-Lorentzian dip model to one ESR contrast spectrum with ``lmfit``.

    ``init_params``: ``{offset, A1, f01, gamma1, ...}`` (optional ``slope``);
    ``None`` → auto-estimate.
    ``linear_bg``: when ``False`` (default) ``slope`` is fixed at 0 and the fit
    is identical to the plain constant-offset model; when ``True`` ``slope``
    varies freely. ``.params`` always contains ``slope`` either way.
    Returns an ``lmfit`` ``MinimizerResult`` with ``.params`` and ``.best_fit``.
    """
    from lmfit import minimize, Parameters

    p0 = init_params if init_params is not None else _auto_init(freq_GHz, spectrum, n_peaks)

    lm_params = Parameters()
    lm_params.add("offset", value=p0["offset"])
    lm_params.add("slope", value=(p0.get("slope", 0.0) if linear_bg else 0.0), vary=linear_bg)
    for i in range(1, n_peaks + 1):
        lm_params.add(f"A{i}", value=p0[f"A{i}"], min=0, max=1.0)
        lm_params.add(f"f0{i}", value=p0[f"f0{i}"],
                      min=float(freq_GHz[0]), max=float(freq_GHz[-1]))
        lm_params.add(f"gamma{i}", value=p0[f"gamma{i}"], min=1e-4, max=0.5)

    def _residual(pars, freq, data):
        r = pars["offset"].value + pars["slope"].value * freq
        for i in range(1, n_peaks + 1):
            A = pars[f"A{i}"].value
            f0 = pars[f"f0{i}"].value
            g = pars[f"gamma{i}"].value
            r = r - A * (g / 2) ** 2 / ((freq - f0) ** 2 + (g / 2) ** 2)
        return r - data

    result = minimize(_residual, lm_params, args=(freq_GHz, spectrum), method="leastsq")
    result.best_fit = spectrum + result.residual
    return result


# ============================================================
# Public API — ROI, per pixel
# ============================================================

def fit_roi_cpu(contrast, mask, freq_GHz, n_peaks=2, n_workers=None, init_params=None,
                linear_bg=False, per_pixel_init=False, init_smooth_n=1):
    """Per-pixel NV ESR fitting over a ROI using CPU multiprocessing (``scipy``).

    ``contrast``: ``(ny, nx, n_freq)`` dF/F array.
    ``init_params``: ``{offset, A1, f01, gamma1, ...}`` (optional ``slope``);
    ``None`` → auto-estimate from the ROI average.
    ``linear_bg``: ``False`` (default) fits a constant offset only, identical to
    before; ``True`` adds a free ``slope·f`` term.
    ``per_pixel_init``: ``False`` (default) fits every pixel from the same
    shared init (``init_params``, or auto-estimated once from the ROI
    average). ``True`` ignores ``init_params`` and instead auto-estimates each
    pixel's init independently from its own spectrum's deepest minima (same
    heuristic as :func:`init_from_spectrum`) — more robust when the dip
    position varies a lot across the ROI, at the cost of losing a
    user-tuned/loaded init.
    ``init_smooth_n``: passed to the auto-init's peak search (see
    :func:`init_from_spectrum`'s ``smooth_n``) — only matters when auto-init
    actually runs (``init_params is None``, or ``per_pixel_init=True``).
    Returns a dict of 2D parameter maps, NaN outside ``mask`` (includes a
    ``slope`` map, all zeros when ``linear_bg`` is False).
    """
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm

    from .roi import roi_spectra

    spectra = roi_spectra(contrast, mask)
    lo, hi = _param_bounds(n_peaks, linear_bg, freq_GHz)

    if per_pixel_init:
        args_list = [
            (spectra[j], freq_GHz,
             _p0_vector(_auto_init(freq_GHz, spectra[j], n_peaks, init_smooth_n), n_peaks, linear_bg),
             (lo, hi))
            for j in range(len(spectra))
        ]
    else:
        avg = spectra.mean(axis=0)
        p0_dict = init_params if init_params is not None else _auto_init(freq_GHz, avg, n_peaks, init_smooth_n)
        p0 = _p0_vector(p0_dict, n_peaks, linear_bg)
        args_list = [(spectra[j], freq_GHz, p0, (lo, hi)) for j in range(len(spectra))]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(
            executor.map(_fit_pixel_worker, args_list),
            total=len(spectra), desc="Fitting pixels (CPU)", **tqdm_dict,
        ))

    return _build_param_maps(
        np.array(results), mask, contrast.shape[:2], n_peaks, freq_GHz, spectra,
        linear_bg=linear_bg,
    )


def fit_roi_gpu(
    contrast, mask, freq_GHz, n_peaks=2, n_iter=1000, lr=8e-3,
    init_params=None, device=None, dtype=None, linear_bg=False, per_pixel_init=False,
    init_smooth_n=1,
):
    """Per-pixel NV ESR fitting over a ROI using batched PyTorch Adam on the GPU.

    All ROI pixels are optimised simultaneously in one batched forward pass.
    Uses CUDA when available (falls back to CPU otherwise).

    ``contrast``: ``(ny, nx, n_freq)`` dF/F array.
    ``init_params``: ``{offset, A1, f01, gamma1, ...}`` (optional ``slope``);
    ``None`` → auto-estimate from the ROI average.
    ``linear_bg``: ``False`` (default) holds ``slope`` at 0 (identical to the
    plain constant-offset fit); ``True`` optimises a ``slope·f`` term too.
    ``per_pixel_init``: ``False`` (default) starts every pixel from the same
    shared init (``init_params``, or auto-estimated once from the ROI
    average). ``True`` ignores ``init_params`` and instead auto-estimates each
    pixel's starting point independently from its own spectrum's deepest
    minima — more robust when the dip position varies a lot across the ROI.
    ``init_smooth_n``: passed to the auto-init's peak search (see
    :func:`init_from_spectrum`'s ``smooth_n``) — only matters when auto-init
    actually runs (``init_params is None``, or ``per_pixel_init=True``).
    ``device``: explicit ``torch.device``; default picks ``cuda`` if available.
    ``dtype``: default ``torch.float32`` (ample for this optimisation).
    Returns a dict of 2D parameter maps, NaN outside ``mask`` (includes a
    ``slope`` map, all zeros when ``linear_bg`` is False).
    """
    import torch

    from .roi import roi_spectra

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float32

    spectra = roi_spectra(contrast, mask)
    N, n_f = spectra.shape

    freq_t = torch.tensor(np.asarray(freq_GHz), dtype=dtype, device=device)
    data_t = torch.tensor(spectra, dtype=dtype, device=device)

    dip0 = 2 if linear_bg else 1  # first dip-parameter column

    if per_pixel_init:
        init_matrix = [
            _p0_vector(_auto_init(freq_GHz, spectra[j], n_peaks, init_smooth_n), n_peaks, linear_bg)
            for j in range(N)
        ]
        init_t = torch.tensor(init_matrix, dtype=dtype, device=device)  # (N, n_params)
        params = torch.nn.Parameter(init_t.clone())
    else:
        avg = spectra.mean(axis=0)
        p0_dict = init_params if init_params is not None else _auto_init(freq_GHz, avg, n_peaks, init_smooth_n)
        init_vals = _p0_vector(p0_dict, n_peaks, linear_bg)
        init_t = torch.tensor(init_vals, dtype=dtype, device=device)
        params = torch.nn.Parameter(init_t.unsqueeze(0).expand(N, -1).clone())

    optimizer = torch.optim.Adam([params], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iter, eta_min=lr / 100)
    freq_unsq = freq_t.unsqueeze(0)  # (1, n_freq) for broadcasting with (N, 1)

    f_min, f_max = float(freq_GHz[0]), float(freq_GHz[-1])

    for _ in range(n_iter):
        optimizer.zero_grad()
        if linear_bg:
            pred = params[:, 0:1] + params[:, 1:2] * freq_unsq
        else:
            pred = params[:, 0:1].expand(-1, n_f)
        for i in range(n_peaks):
            A = params[:, dip0 + 3 * i : dip0 + 1 + 3 * i]      # (N, 1)
            f0 = params[:, dip0 + 1 + 3 * i : dip0 + 2 + 3 * i]  # (N, 1)
            gamma = params[:, dip0 + 2 + 3 * i : dip0 + 3 + 3 * i]  # (N, 1)
            pred = pred - A * (gamma / 2) ** 2 / (
                (freq_unsq - f0) ** 2 + (gamma / 2) ** 2
            )
        loss = ((pred - data_t) ** 2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            for i in range(n_peaks):
                params.data[:, dip0 + 3 * i].clamp_(min=0)
                params.data[:, dip0 + 1 + 3 * i].clamp_(f_min, f_max)
                params.data[:, dip0 + 2 + 3 * i].clamp_(min=1e-4)

    params_np = params.detach().cpu().numpy().astype(np.float64)
    return _build_param_maps(
        params_np, mask, contrast.shape[:2], n_peaks, freq_GHz, spectra,
        linear_bg=linear_bg,
    )


def fit_roi(contrast, mask, freq_GHz, n_peaks=2, use_gpu="auto", init_params=None,
            linear_bg=False, x_bin=1, y_bin=1, bin_mode="block", per_pixel_init=False,
            init_smooth_n=1):
    """Fit NV ESR spectra for all pixels within ``mask``.

    ``use_gpu``: ``'auto'`` (prefer CUDA if available) | ``True`` | ``False``.
    ``init_params``: ``{offset, A1, f01, gamma1, ...}`` (optional ``slope``);
    ``None`` → auto-estimate.
    ``linear_bg``: ``False`` (default) → constant-offset fit, unchanged; ``True``
    → adds a free ``slope·f`` linear background.
    ``x_bin``/``y_bin`` + ``bin_mode``: trade spatial resolution for SNR before
    fitting (default ``x_bin=y_bin=1`` is an exact no-op regardless of
    ``bin_mode``). Two modes, see :mod:`widefieldesr.binning`:
      - ``bin_mode="block"`` (default): non-overlapping ``x_bin``×``y_bin``
        blocks, one fit per block, replicated back onto every raw pixel the
        block covers. Fewer, better-SNR fits (faster); output is "blocky"
        (piecewise-constant per block).
      - ``bin_mode="sliding"``: a ``x_bin``×``y_bin`` moving-window average
        centred on *every* pixel, so every pixel is still fit individually
        (no speedup) but from a smoothed, higher-SNR spectrum — output is full
        resolution with no blockiness.
    Either mode returns maps of full ``contrast.shape[:2]``, NaN outside
    ``mask``.
    ``per_pixel_init``: ``False`` (default) — every pixel (or, with block
    binning, every block) starts from the same shared init: ``init_params`` if
    given, else auto-estimated once from the ROI average. ``True`` ignores
    ``init_params`` and instead auto-estimates each pixel's/block's own init
    independently from its own spectrum's deepest minima. This can help when
    the dip position varies a lot across the ROI (a single shared seed can't
    be close for every pixel), but trades away the shared-seed's robustness:
    an individual pixel's spectrum is noisier than the ROI average, so its own
    peak search can lock onto a spurious minimum and the fit then converges
    to the wrong dip pair for that pixel. Binning above raises the SNR per
    fitted spectrum and is the natural pairing with this option; even so,
    always sanity-check a per-pixel-init map's median/spread against the
    shared-init result before trusting it.
    ``init_smooth_n``: smooths a copy of each spectrum before the auto-init
    peak search only (never the data actually fit) — see
    :func:`init_from_spectrum`'s ``smooth_n``. Only matters when auto-init
    actually runs (``init_params is None``, or ``per_pixel_init=True``); the
    natural pairing with ``per_pixel_init=True``, since that is exactly the
    case where each individual (noisier) spectrum's own peak search benefits
    from smoothing.
    Returns a dict of 2D parameter maps (NaN outside ``mask``):
    ``f0_i, A_i, gamma_i`` for ``i`` in ``1..n_peaks``, plus ``offset``,
    ``slope`` (zeros if ``linear_bg`` is False), ``redchi``.
    """
    if bin_mode == "sliding" and (x_bin != 1 or y_bin != 1):
        from .binning import smooth_contrast

        contrast = smooth_contrast(contrast, mask, y_bin, x_bin)
        x_bin = y_bin = 1  # already averaged; fall through to a per-pixel fit

    if x_bin != 1 or y_bin != 1:
        from .binning import bin_contrast, bin_mask, upsample_maps

        coarse_mask = bin_mask(mask, y_bin, x_bin)
        coarse_contrast = bin_contrast(contrast, mask, y_bin, x_bin)
        coarse_maps = fit_roi(coarse_contrast, coarse_mask, freq_GHz, n_peaks=n_peaks,
                              use_gpu=use_gpu, init_params=init_params, linear_bg=linear_bg,
                              per_pixel_init=per_pixel_init, init_smooth_n=init_smooth_n)
        return upsample_maps(coarse_maps, y_bin, x_bin, contrast.shape[:2], mask)

    import torch
    prefer_gpu = use_gpu is True or (
        use_gpu == "auto" and torch.cuda.is_available()
    )
    if prefer_gpu:
        return fit_roi_gpu(contrast, mask, freq_GHz, n_peaks=n_peaks,
                           init_params=init_params, linear_bg=linear_bg,
                           per_pixel_init=per_pixel_init, init_smooth_n=init_smooth_n)
    return fit_roi_cpu(contrast, mask, freq_GHz, n_peaks=n_peaks,
                       init_params=init_params, linear_bg=linear_bg,
                       per_pixel_init=per_pixel_init, init_smooth_n=init_smooth_n)
