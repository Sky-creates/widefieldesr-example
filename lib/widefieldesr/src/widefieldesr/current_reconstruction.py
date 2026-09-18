"""2D sheet-current reconstruction from a measured out-of-plane field map.

Fourier-domain inversion of the 2D Biot-Savart law: given ``Bz(x, y)`` at a
known sensor standoff, recover the 2D sheet current density ``(jx, jy)`` that
produced it. Model-agnostic -- no assumption about *why* the current exists
(superconducting screening, a normal-metal device, anything else), only that
it can be described as an in-plane current confined to a well-defined 2D
layer (exact for a genuine thin film; a good approximation for a bulk
superconductor's near-surface Meissner screening current, since that's
confined to within one London penetration depth of the surface -- negligible
next to typical imaged feature sizes).

Method: Roth, B. J., Sepulveda, N. G. & Wikswo, J. P. "Using a magnetometer
to image a two-dimensional current distribution." J. Appl. Phys. 65, 361-372
(1989), https://doi.org/10.1063/1.342549. Adapted from the reference
implementation at https://github.com/loganbvh/current-reconstruction (MIT
license, Logan Bishop-Van Horn) -- ported to plain-array arguments (no
bespoke ``Image`` wrapper) to match this package's convention (see
:mod:`widefieldesr.binning`, :func:`widefieldesr.fitting.fit_roi`: arrays +
explicit scalar parameters, never wrapper objects), and with pixel size an
explicit caller-supplied argument rather than a lib-level constant (physical
pixel size is project/setup-specific, not general physics -- see this
workspace's ``constants.PIXEL_SIZE_UM`` convention).

The deconvolution divides by ``K = sqrt(kx^2 + ky^2)`` in Fourier space, which
is singular at the DC (zero-frequency) bin. This is avoided, not patched
around: the k-space grid is built from an odd-length array (trimming the last
row/column of an even-sized input first, same as the reference
implementation), so ``k=0`` is never one of the sampled frequencies -- see
:func:`_trim_to_odd`.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import mu_0

__all__ = ["reconstruct_current"]


def _hanning_2d(kx, ky, kx_max, ky_max):
    """2D Hanning window, Eq. 18 in J. Appl. Phys. 65, 361 (1989)."""
    Kx, Ky = np.meshgrid(kx, ky)
    k = np.sqrt(Kx**2 + Ky**2)
    kmax = np.sqrt(kx_max**2 + ky_max**2)
    window = 0.5 * (1 + np.cos(np.pi * k / kmax))
    window[k > kmax] = 0.0
    return window


def _trim_to_odd(arr):
    """Drop the last row/column of an even axis -- keeps k=0 off the FFT grid."""
    ny, nx = arr.shape
    if ny % 2 == 0:
        arr = arr[:-1, :]
    if nx % 2 == 0:
        arr = arr[:, :-1]
    return arr


def reconstruct_current(
    Bz,
    pixel_size_um,
    standoff_um,
    psf=None,
    kx_max=1.0,
    ky_max=1.0,
    x_pad=None,
    y_pad=None,
    pad_mode="linear_ramp",
):
    """Reconstruct the 2D sheet current density from a Bz map.

    Parameters
    ----------
    Bz : (ny, nx) array
        Local out-of-plane field, **Tesla**, on a uniform pixel grid.
    pixel_size_um : float
        Physical pixel size, micrometres.
    standoff_um : float
        Sensor-to-source-plane separation, micrometres, same units as
        ``pixel_size_um``.
    psf : (ny, nx) array, optional
        Sensor point-spread function, centred (peak at the array's middle),
        same shape as ``Bz``. ``None`` (default) treats ``Bz`` as the field
        itself rather than a PSF-convolved flux signal.
    kx_max, ky_max : float
        Hanning-window cutoff spatial frequencies, 1/um. Smaller values
        filter more aggressively -- the deconvolution's division by
        ``1/exp(-K*standoff)`` amplifies high-k noise, so this is the
        regularization knob.
    x_pad, y_pad : int, optional
        Columns/rows of edge padding before the FFT (avoids periodic-wraparound
        artifacts). Default ``len(axis) // 2`` on each side.
    pad_mode : str
        ``numpy.pad`` mode for ``Bz``; default ``"linear_ramp"``. ``psf`` (if
        given) is always zero-padded, since a point-spread function should
        taper to zero away from its support.

    Returns
    -------
    jx, jy : arrays
        In-plane sheet current density components, **A/m**. Shape matches
        ``Bz`` exactly, UNLESS ``Bz`` had an even dimension along either
        axis, in which case that axis is one pixel smaller (the last
        row/column is dropped before the FFT to keep the k=0 bin off the
        sampled grid -- see the module docstring).
    """
    Bz = _trim_to_odd(np.asarray(Bz, dtype=float))
    ny, nx = Bz.shape

    if x_pad is None:
        x_pad = nx // 2
    if y_pad is None:
        y_pad = ny // 2

    Bz_padded = np.pad(Bz, ((y_pad, y_pad), (x_pad, x_pad)), mode=pad_mode)
    mag_k = np.fft.fftshift(np.fft.fft2(Bz_padded))

    if psf is None:
        psf_k = 1.0
    else:
        psf = _trim_to_odd(np.asarray(psf, dtype=float))
        if psf.shape != (ny, nx):
            raise ValueError(
                f"psf shape {psf.shape} must match the (odd-trimmed) Bz shape {(ny, nx)}"
            )
        psf_padded = np.pad(psf, ((y_pad, y_pad), (x_pad, x_pad)), mode="constant")
        psf_k = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(psf_padded)))

    # mu_0 is SI (T*m/A) -- every length feeding the physics below must be in
    # METERS, not microns, or the result is silently off by a power of 1e6.
    dx_m = dy_m = float(pixel_size_um) * 1e-6
    standoff_m = float(standoff_um) * 1e-6
    ny_p, nx_p = Bz_padded.shape
    kx = np.linspace(-np.pi / dx_m, np.pi / dx_m, nx_p, endpoint=False)
    ky = np.linspace(-np.pi / dy_m, np.pi / dy_m, ny_p, endpoint=False)
    Kx, Ky = np.meshgrid(kx, ky)
    K = np.sqrt(Kx**2 + Ky**2)

    # kx_max/ky_max are documented in 1/um; convert to 1/m to match Kx, Ky.
    window = _hanning_2d(kx, ky, kx_max * 1e6, ky_max * 1e6)

    factor = 2j * mag_k * window / (mu_0 * np.exp(-K * standoff_m) * K * psf_k)
    jx_k = -Ky * factor
    jy_k = +Kx * factor

    jx = np.fft.ifft2(np.fft.ifftshift(jx_k)).real
    jy = np.fft.ifft2(np.fft.ifftshift(jy_k)).real

    # `arr[pad:-pad]` is WRONG when pad == 0 (-0 == 0, giving an empty slice,
    # not "no crop") -- use None as the upper bound in that case.
    y_hi = -y_pad if y_pad else None
    x_hi = -x_pad if x_pad else None
    jx = jx[y_pad:y_hi, x_pad:x_hi]
    jy = jy[y_pad:y_hi, x_pad:x_hi]

    return jx, jy
