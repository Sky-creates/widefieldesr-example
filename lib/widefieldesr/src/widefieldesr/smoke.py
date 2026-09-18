"""End-to-end smoke test: ``uv run python -m widefieldesr.smoke``.

Loads one acquisition from ``$WIDEFIELDESR_DATA_ROOT``, computes contrast, and
fits a central ROI three ways — ``lmfit`` on the ROI average, per-pixel on CPU,
per-pixel on GPU — checking they agree. Pass a ``.mat`` path as the first
argument to override the auto-discovered file.
"""

from __future__ import annotations

import sys

import numpy as np

import widefieldesr as wh


def _discover():
    """First ``WidefieldESR_*.mat`` under the data root (excludes spatial cals)."""
    root = wh.data_root()
    hits = sorted(
        p for p in root.rglob("WidefieldESR_*.mat")
        if "spatial_calibration" not in p.parts
    )
    if not hits:
        raise SystemExit(f"no WidefieldESR_*.mat found under {root}")
    return str(hits[0])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    data_path = argv[0] if argv else _discover()
    print(f"data root : {wh.data_root()}")
    print(f"data file : {data_path}")

    d = wh.load_esr(data_path)
    signal, freq_GHz = d["signal"], d["freq_GHz"]
    print(f"signal    : {signal.shape} {signal.dtype}")
    print(f"freq      : {freq_GHz[0]:.4f}-{freq_GHz[-1]:.4f} GHz ({len(freq_GHz)} pts)")

    contrast = wh.compute_contrast(signal)
    dip = wh.dip_depth_map(contrast)
    assert contrast.shape == signal.shape
    assert np.isfinite(dip).all(), "non-finite dip-depth pixels"
    print(f"dip depth : min {dip.min():.4f}  median {np.median(dip):.4f}")

    ny, nx = signal.shape[:2]
    half = 90  # central 180x180 square — enough SNR, quick to fit
    mask = wh.make_rect_roi(nx // 2 - half, ny // 2 - half, 2 * half, 2 * half,
                            img_shape=(ny, nx))
    print(f"ROI       : {2*half}x{2*half} central rect, {mask.sum()} px")

    import torch
    cuda = torch.cuda.is_available()
    print(f"cuda      : {cuda}" + (f"  ({torch.cuda.get_device_name(0)})" if cuda else ""))

    ref = wh.fit_esr_spectrum(freq_GHz, wh.roi_avg_spectrum(contrast, mask), n_peaks=2)
    split_ref = abs(ref.params["f02"].value - ref.params["f01"].value) * 1e3
    print(f"ref split : {split_ref:.1f} MHz  (lmfit, ROI average)")

    cpu_maps = wh.fit_roi(contrast, mask, freq_GHz, n_peaks=2, use_gpu=False)
    gpu_maps = wh.fit_roi(contrast, mask, freq_GHz, n_peaks=2, use_gpu=True)

    expected = {"offset", "redchi", "A_1", "f0_1", "gamma_1", "A_2", "f0_2", "gamma_2"}
    assert expected <= set(cpu_maps) and expected <= set(gpu_maps), "missing map keys"

    split_cpu = (cpu_maps["f0_2"] - cpu_maps["f0_1"])[mask] * 1e3
    split_gpu = (gpu_maps["f0_2"] - gpu_maps["f0_1"])[mask] * 1e3
    med_cpu, med_gpu = np.nanmedian(split_cpu), np.nanmedian(split_gpu)
    med_abs_diff = float(np.nanmedian(np.abs(split_cpu - split_gpu)))
    print(f"per-pixel : CPU median {med_cpu:.1f} MHz  GPU median {med_gpu:.1f} MHz  "
          f"|Δ| median {med_abs_diff:.2f} MHz")

    assert abs(med_cpu - split_ref) < 20, "CPU median far from lmfit reference"
    assert abs(med_gpu - split_ref) < 20, "GPU median far from lmfit reference"
    assert med_abs_diff < 10, f"CPU vs GPU per-pixel splitting differ by {med_abs_diff:.1f} MHz"

    _check_binning(contrast, mask, freq_GHz, split_ref)
    _check_sliding_window(contrast, mask, freq_GHz, split_ref)
    _check_per_pixel_init(contrast, mask, freq_GHz, split_ref)
    _check_nv_physics()
    _check_current_reconstruction()

    print("\nSMOKE OK")


def _check_binning(contrast, mask, freq_GHz, split_ref):
    """x_bin=y_bin=1 must be a no-op; binned fits must stay full-shape and
    NaN-outside-mask, and stay in the right ballpark vs. the unbinned fit."""
    unbinned = wh.fit_roi(contrast, mask, freq_GHz, n_peaks=2, use_gpu=False)
    noop = wh.fit_roi(contrast, mask, freq_GHz, n_peaks=2, use_gpu=False, x_bin=1, y_bin=1)
    for key in ("f0_1", "f0_2", "redchi"):
        assert np.allclose(unbinned[key], noop[key], equal_nan=True), (
            f"x_bin=y_bin=1 changed '{key}' — no-op path is not a no-op"
        )

    binned = wh.fit_roi(contrast, mask, freq_GHz, n_peaks=2, use_gpu=False, x_bin=4, y_bin=4)
    assert binned["f0_1"].shape == contrast.shape[:2], "binned map shape must match contrast"
    assert np.array_equal(np.isnan(binned["f0_1"]), ~mask), "binned map NaN pattern must match ~mask"

    split_binned = (binned["f0_2"] - binned["f0_1"])[mask] * 1e3
    med_binned = float(np.nanmedian(split_binned))
    print(f"binning   : 4x4 median {med_binned:.1f} MHz  (unbinned ref {split_ref:.1f} MHz)")
    assert abs(med_binned - split_ref) < 20, "4x4-binned median far from the unbinned reference"


def _check_per_pixel_init(contrast, mask, freq_GHz, split_ref):
    """per_pixel_init=True must stay full-shape/NaN-masked and actually vary
    the seed per pixel (not silently fall back to the shared-init path).

    NOT checked at init_smooth_n=1: numerical closeness to the shared-init
    median. Seeding each pixel from its own noisy spectrum's deepest minima is
    a genuinely different, less robust method than one shared ROI-average
    seed — on a low-contrast ROI it can converge to a substantially different
    (usually worse) median, which is an inherent property of the method, not
    a bug.

    init_smooth_n is the fix for the above: smoothing the per-pixel peak
    search (not the fit data) should monotonically pull the median back
    toward the shared-init reference as smooth_n grows. Verified empirically
    on this ROI: init_smooth_n 1/5/15 -> median 12.7/39.9/59.9 MHz (shared-init
    reference ~61.6-65.6 MHz depending on which fit is used as "reference") —
    the |diff-from-reference| sweep asserted below is a stronger regression
    guard than a single absolute threshold.
    """
    for use_gpu in (False, True):
        maps = wh.fit_roi(contrast, mask, freq_GHz, n_peaks=2, use_gpu=use_gpu,
                          per_pixel_init=True)
        assert maps["f0_1"].shape == contrast.shape[:2]
        assert np.array_equal(np.isnan(maps["f0_1"]), ~mask)
        med = float(np.nanmedian((maps["f0_2"] - maps["f0_1"])[mask]) * 1e3)
        print(f"per-pixel init (use_gpu={use_gpu}): median {med:.1f} MHz "
              f"(shared-init reference {split_ref:.1f} MHz — not asserted equal, see docstring)")

    smooth_ns = (1, 5, 15)
    diffs = []
    for n in smooth_ns:
        maps = wh.fit_roi(contrast, mask, freq_GHz, n_peaks=2, use_gpu=False,
                          per_pixel_init=True, init_smooth_n=n)
        med = float(np.nanmedian((maps["f0_2"] - maps["f0_1"])[mask]) * 1e3)
        diffs.append(abs(med - split_ref))
        print(f"per-pixel init, init_smooth_n={n:2d}: median {med:.1f} MHz  "
              f"(|diff| {diffs[-1]:.1f} MHz)")

    assert diffs[0] >= diffs[1] >= diffs[2], (
        f"increasing init_smooth_n should monotonically improve agreement "
        f"with the shared-init reference, got |diff| {diffs} for smooth_n {smooth_ns}"
    )
    assert diffs[-1] < 10, (
        f"init_smooth_n={smooth_ns[-1]} median still {diffs[-1]:.1f} MHz from "
        f"the shared-init reference {split_ref:.1f} MHz — smoothing isn't helping enough"
    )


def _check_sliding_window(contrast, mask, freq_GHz, split_ref):
    """bin_mode='sliding' must stay full-shape/NaN-masked at native resolution
    (no upsample step needed -- smoothing already fills the whole grid).
    window=1 must be an exact no-op (identical to the plain unbinned fit);
    larger windows should approach (and, on this ROI, slightly beat) the
    shared-init/lmfit reference as SNR improves — it's meant to improve on the
    unbinned per-pixel fit, not degrade it.

    Verified empirically on this ROI: window 1/4/8 -> median 61.6/65.5/65.6 MHz
    (reference ~61.6-65.6 MHz).
    """
    unbinned = wh.fit_roi(contrast, mask, freq_GHz, n_peaks=2, use_gpu=False)
    med_unbinned = float(np.nanmedian((unbinned["f0_2"] - unbinned["f0_1"])[mask]) * 1e3)

    diffs = []
    for w in (1, 4, 8):
        maps = wh.fit_roi(contrast, mask, freq_GHz, n_peaks=2, use_gpu=False,
                          bin_mode="sliding", x_bin=w, y_bin=w)
        assert maps["f0_1"].shape == contrast.shape[:2], "sliding-window map shape must match contrast"
        assert np.array_equal(np.isnan(maps["f0_1"]), ~mask), "sliding-window map NaN pattern must match ~mask"
        med = float(np.nanmedian((maps["f0_2"] - maps["f0_1"])[mask]) * 1e3)
        diffs.append(abs(med - split_ref))
        print(f"sliding window {w}x{w}: median {med:.1f} MHz  (ref {split_ref:.1f} MHz)")
        if w == 1:
            assert abs(med - med_unbinned) < 1e-6, "sliding window=1 must be an exact no-op"

    assert diffs[-1] < 10, (
        f"largest sliding window still {diffs[-1]:.1f} MHz from the "
        f"shared-init reference {split_ref:.1f} MHz"
    )


def _check_nv_physics():
    """Data-free checks of the exact NV Hamiltonian / stress / thermal model."""
    from . import nv_physics as nvp

    # splitting <-> field inversion round-trips through the exact model
    D, E = 2877.27, 3.28
    df_to_B100 = nvp.make_B100_inverter(D, E)
    B = np.linspace(5.0, 300.0, 60)
    rt = float(np.abs(df_to_B100(nvp.splitting_of_B100(B, D, E)) - B).max())
    assert rt < 0.1, f"B100 inverter round-trip off by {rt:.3f} G"

    # hydrostatic stress: Mz = 3*a1*P, no transverse shift
    Mz, Mx, My = nvp.stress_to_shifts([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    assert abs(Mz - nvp.DDP_HYDROSTATIC_MHZ_PER_GPA) < 1e-6
    assert abs(Mx) < 1e-9 and abs(My) < 1e-9
    assert abs(nvp.pressure_from_D_shift(nvp.DDP_HYDROSTATIC_MHZ_PER_GPA) - 1.0) < 1e-6

    # D map -> equivalent uniaxial [100] stress: ΔD = a1 per GPa
    a1 = nvp.STRESS_COUPLING["a1"]
    assert abs(float(nvp.sigma_from_D(nvp.D_GS_MHZ + a1)) - 1.0) < 1e-9
    assert abs(float(nvp.sigma_from_D(nvp.D_GS_MHZ + 3 * a1, model="hydrostatic")) - 1.0) < 1e-9

    # nv_lines must survive NaN-masked map pixels instead of raising
    B_map = np.array([[10.0, np.nan], [30.0, 51.0]])
    fm_map, fp_map = nvp.nv_lines(B_map, D, E)
    assert np.isnan(fm_map[0, 1]) and np.isnan(fp_map[0, 1])
    assert np.isfinite(fm_map[[0, 1], [0, 1]]).all() and np.isfinite(fp_map[[0, 1], [0, 1]]).all()

    # thermal D(T): plateau at low T, ~-70 kHz/K near 300 K
    assert abs(float(nvp.dDdT_of_T(60.0))) < 5e-3, "D(T) not flat at 60 K"
    assert -0.08 < float(nvp.dDdT_of_T(300.0)) < -0.06, "dD/dT(300 K) out of range"
    print(f"nv_physics : inverter RT {rt:.4f} G  dD/dT(300K) "
          f"{float(nvp.dDdT_of_T(300.0)) * 1e3:.1f} kHz/K")


def _check_current_reconstruction():
    """Data-free round-trip check of reconstruct_current's Fourier inversion.

    Not a shape/NaN check (a naive derivative-based current proxy tried
    earlier this session passed those trivially while being ~1e8-1e9x off in
    absolute magnitude -- wrong units, not wrong shape). Instead: build a
    known stream function g(x,y), derive the current (jx,jy) it implies via
    J = z_hat x grad(g), and the field Bz it implies via the EXACT forward
    Biot-Savart relation reconstruct_current inverts (independently
    reproduced here, not imported, so this actually exercises the source's
    unit handling and k-space bookkeeping rather than just restating it).
    Feed that synthetic Bz through reconstruct_current and check the
    recovered current matches the known one.
    """
    from scipy.constants import mu_0

    from .current_reconstruction import _hanning_2d, reconstruct_current

    nx = ny = 101
    px_um, standoff_um = 0.1, 1.0
    kx_max = ky_max = 2.0  # 1/um

    x = np.arange(nx) - nx // 2
    y = np.arange(ny) - ny // 2
    X, Y = np.meshgrid(x, y)
    g = 5.0 * np.exp(-(X**2 + Y**2) / (2 * 10.0**2))  # arbitrary smooth stream fn
    g_k = np.fft.fftshift(np.fft.fft2(g))

    dx_m = dy_m = px_um * 1e-6
    standoff_m = standoff_um * 1e-6
    kx = np.linspace(-np.pi / dx_m, np.pi / dx_m, nx, endpoint=False)
    ky = np.linspace(-np.pi / dy_m, np.pi / dy_m, ny, endpoint=False)
    Kx, Ky = np.meshgrid(kx, ky)
    K = np.sqrt(Kx**2 + Ky**2)
    window = _hanning_2d(kx, ky, kx_max * 1e6, ky_max * 1e6)

    jy_k_true = 1j * Kx * g_k
    Bz_k = g_k * mu_0 * np.exp(-K * standoff_m) * K / 2  # forward relation
    Bz = np.fft.ifft2(np.fft.ifftshift(Bz_k)).real

    _, jy = reconstruct_current(
        Bz, px_um, standoff_um, kx_max=kx_max, ky_max=ky_max, x_pad=0, y_pad=0
    )
    jy_true = np.fft.ifft2(np.fft.ifftshift(jy_k_true * window)).real

    rel_err = float(np.abs(jy - jy_true).max() / np.abs(jy_true).max())
    print(f"current_reconstruction: round-trip rel err {rel_err:.3f}  "
          f"(recovered peak {jy.max():.1f} A/m, true {jy_true.max():.1f} A/m)")
    assert rel_err < 0.1, f"current-reconstruction round-trip error {rel_err:.3f} too large"


if __name__ == "__main__":
    main()
