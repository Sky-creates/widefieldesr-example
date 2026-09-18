"""nv_physics — NV⁻ ground-state spin Hamiltonian and its field / stress / thermal shifts.

Everything here is the *exact* S=1 electronic ground-state model of the NV⁻
centre, evaluated by diagonalising the 3×3 Hamiltonian (no perturbative
splitting formula). Units throughout:

* energies / frequencies : **MHz**
* magnetic field         : **gauss**
* stress                 : **GPa**
* temperature            : **kelvin**

The Hamiltonian (in the NV frame, ``|m_s = 0, ±1⟩`` basis) is

    H = (D + M_z)·S_z²
      + E·(S_x² − S_y²)
      + M_x·(S_y² − S_x²) + M_y·(S_x S_y + S_y S_x)      # crystal stress
      + γ_e·(B_⊥·S_x + B_∥·S_z)                          # Zeeman

with ``γ_e = 2.8025`` MHz/G.  ``D`` is the zero-field splitting, ``E`` the
transverse strain / electric-field parameter, and ``(M_z, M_x, M_y)`` the
stress-induced shifts.  Note the sign convention: ``E`` multiplies
``(S_x² − S_y²)`` while the Barson stress term ``M_x`` multiplies the opposite
combination ``(S_y² − S_x²)`` — this matches Barson et al. and the reference
implementation in ``AnmayG/peak_finder`` (``params_to_peaks.m``).  Use
:func:`effective_DE` to get the ``(D_eff, E_eff)`` a fit would actually report.

Terminology — "splitting" is overloaded in the NV literature, so pin it down:
``D`` is *the* zero-field splitting (standard usage); the two ODMR dip
frequencies ``f₋, f₊`` returned by :func:`nv_lines` are never called "the
splitting" bare — their difference is ``Δf = f₊ − f₋`` (what
:func:`splitting_of_B100` computes and what a field-calibration fit measures),
and their mean ``c = (f₋+f₊)/2`` is the line centre.  ``E`` is a fit
*parameter*, not a measured splitting, and stress/thermal changes to it are
written ``ΔE`` (never a bare "shift" that could be misread as ``Δf``).

References
----------
* Zeeman / [100]-cut model: as used across the ``bi2212-widefield-esr`` notebooks.
* Stress coupling: M. S. Barson et al., "Nanomechanical Sensing Using Spins in
  Diamond", Nano Lett. **17**, 1496 (2017); combinations as implemented in
  ``AnmayG/peak_finder``.  Hydrostatic limit ``dD/dP = 3·a₁ = 14.58`` MHz/GPa,
  consistent with M. W. Doherty et al., PRL **112**, 047601 (2014).
* Thermal ``D(T)``: default is the two-phonon-mode model of M. C. Cambria et
  al., PRB **108**, L180102 (2023) [arXiv:2306.05318], valid 15–500 K and
  non-divergent; the Chen et al. (APL **99**, 161903 (2011)) 5th-order
  polynomial is also provided.
* Stark: E. van Oort & M. Glasbeek, Chem. Phys. Lett. **168**, 529 (1990);
  d_∥ ≈ 0.35, d_⊥ ≈ 17 Hz·cm/V.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "GAMMA_MHZ_PER_G",
    "NV_PROJECTION",
    "D_GS_MHZ",
    "STRESS_COUPLING",
    "DDP_HYDROSTATIC_MHZ_PER_GPA",
    "STARK_PAR_HZ_CM_PER_V",
    "STARK_PERP_HZ_CM_PER_V",
    "SX",
    "SY",
    "SZ",
    "SX2",
    "SY2",
    "SZ2",
    "nv_hamiltonian",
    "nv_lines",
    "splitting_of_B100",
    "make_B100_inverter",
    "stress_to_shifts",
    "effective_DE",
    "hydrostatic_D_shift",
    "pressure_from_D_shift",
    "sigma_from_D",
    "stress_from_DE",
    "D_of_T",
    "dDdT_of_T",
    "stark_shifts",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
GAMMA_MHZ_PER_G = 2.8025
"""NV⁻ gyromagnetic ratio γ_e / 2π (MHz/G)."""

NV_PROJECTION = 1.0 / np.sqrt(3.0)
"""cos(54.7356°) = 1/√3 — axial projection factor for a [100]-cut diamond."""

D_GS_MHZ = 2870.0
"""Nominal ground-state ZFS near 300 K (unstressed).  See :func:`D_of_T` for T
dependence."""

# Barson et al. 2017, spin–stress coupling constants (MHz/GPa).
STRESS_COUPLING = {"a1": 4.86, "a2": -3.7, "b": -2.3, "c": 3.5}

DDP_HYDROSTATIC_MHZ_PER_GPA = 3.0 * STRESS_COUPLING["a1"]  # = 14.58
"""Hydrostatic dD/dP: falls out of the Barson combination as 3·a₁."""

# Stark coupling (Hz per V/cm) — van Oort & Glasbeek 1990.
STARK_PAR_HZ_CM_PER_V = 0.35
STARK_PERP_HZ_CM_PER_V = 17.0

_KB_MEV_PER_K = 8.617333262e-2  # Boltzmann constant in meV/K

# --------------------------------------------------------------------------- #
# Spin-1 operators (|m_s = +1, 0, −1⟩ basis)
# --------------------------------------------------------------------------- #
SX = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float) / np.sqrt(2)
SY = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=complex) / np.sqrt(2)
SZ = np.diag([1.0, 0.0, -1.0])

SX2 = (SX @ SX).real
SY2 = (SY @ SY).real
SZ2 = SZ @ SZ
_SXY_SYX = (SX @ SY + SY @ SX)  # {S_x, S_y}


# --------------------------------------------------------------------------- #
# Hamiltonian and ODMR lines
# --------------------------------------------------------------------------- #
def nv_hamiltonian(D, E, b_par=0.0, b_perp=0.0, *, Mz=0.0, Mx=0.0, My=0.0,
                   gamma=GAMMA_MHZ_PER_G):
    """Build the S=1 NV ground-state Hamiltonian (MHz) in the NV frame.

    All scalar arguments broadcast against each other; the return shape is
    ``(*broadcast_shape, 3, 3)`` (complex).  ``b_par`` / ``b_perp`` are the
    axial and transverse field components in gauss.  ``Mz, Mx, My`` are the
    Barson stress shifts (see module docstring for the sign convention).
    """
    D, E, b_par, b_perp, Mz, Mx, My = (
        np.asarray(v, dtype=float) for v in (D, E, b_par, b_perp, Mz, Mx, My)
    )
    D, E, b_par, b_perp, Mz, Mx, My = np.broadcast_arrays(
        D, E, b_par, b_perp, Mz, Mx, My
    )
    c = lambda a: np.asarray(a)[..., None, None]  # noqa: E731  broadcast helper
    H = (
        c(D + Mz) * SZ2
        + c(E) * (SX2 - SY2)
        + c(Mx) * (SY2 - SX2)
        + c(My) * _SXY_SYX
        + gamma * (c(b_perp) * SX + c(b_par) * SZ)
    )
    return H


def nv_lines(B100, D, E, ge=GAMMA_MHZ_PER_G):
    """Exact ODMR frequencies ``(f₋, f₊)`` (MHz) for a [100]-cut NV.

    ``B100`` is the total field magnitude (G) along [100]; the cut fixes, for
    every NV family, ``B_∥ = B100/√3`` and ``B_⊥ = B100·√(2/3)``.  Returns two
    arrays shaped like ``np.atleast_1d(B100)``.

    Behaviourally identical to the ``nv_lines`` cell previously copy-pasted into
    the ``bi2212-widefield-esr`` notebooks.  NaN/inf entries in ``B100`` (e.g. a
    masked-bad-pixel map) pass through as NaN in both outputs rather than
    raising ``LinAlgError`` — ``eigvalsh`` cannot diagonalise a NaN matrix.
    """
    b = np.atleast_1d(np.asarray(B100, dtype=float))
    bad = ~np.isfinite(b)
    b_safe = np.where(bad, 0.0, b) if bad.any() else b
    b_par = b_safe / np.sqrt(3.0)
    b_perp = b_safe * np.sqrt(2.0 / 3.0)
    H = nv_hamiltonian(D, E, b_par, b_perp, gamma=ge)  # (..., 3, 3)
    w = np.linalg.eigvalsh(H)  # ascending, shape (..., 3)
    fm = w[..., 1] - w[..., 0]
    fp = w[..., 2] - w[..., 0]
    if bad.any():
        fm = np.where(bad, np.nan, fm)
        fp = np.where(bad, np.nan, fp)
    return fm, fp


def splitting_of_B100(B100, D, E, ge=GAMMA_MHZ_PER_G):
    """``f₊ − f₋`` (MHz) for a [100]-cut NV in total field ``B100`` (G)."""
    fm, fp = nv_lines(B100, D, E, ge)
    return fp - fm


def make_B100_inverter(D, E, ge=GAMMA_MHZ_PER_G, b_max_G=400.0, n=4001):
    """Return ``df_to_B100(df_MHz)`` — the monotonic inverse of
    :func:`splitting_of_B100` for fixed ``D``, ``E``.

    Builds a dense ``[0, b_max_G]`` lookup grid (default 0.1 G spacing) and
    interpolates with :func:`numpy.interp`.  Raises ``ValueError`` if the
    splitting is not strictly increasing over the grid (then it cannot be
    inverted this way).  Replaces the ``_split_of_B100`` + grid + ``df_to_B100``
    block duplicated in the spot-2 notebooks.
    """
    b_grid = np.linspace(0.0, float(b_max_G), int(n))
    df_grid = splitting_of_B100(b_grid, D, E, ge)
    if not np.all(np.diff(df_grid) > 0):
        raise ValueError(
            "splitting(B100) is not strictly monotonic over "
            f"[0, {b_max_G}] G — cannot np.interp-invert"
        )

    def df_to_B100(df_MHz):
        """Splitting (MHz) → total field ∥ [100] (G), via the exact NV model."""
        return np.interp(df_MHz, df_grid, b_grid)

    return df_to_B100


# --------------------------------------------------------------------------- #
# Crystal stress  ←→  (D, E)
# --------------------------------------------------------------------------- #
def _stress_components(stress):
    """Coerce a Voigt 6-vector ``[σxx, σyy, σzz, σyz, σxz, σxy]`` or a symmetric
    3×3 matrix (GPa, NV frame) to ``(xx, yy, zz, yz, xz, xy)``."""
    s = np.asarray(stress, dtype=float)
    if s.shape == (3, 3):
        return s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]
    if s.shape == (6,):
        return s[0], s[1], s[2], s[3], s[4], s[5]
    raise ValueError(
        "stress must be a length-6 Voigt vector [xx,yy,zz,yz,xz,xy] "
        f"or a 3x3 matrix, got shape {s.shape}"
    )


def stress_to_shifts(stress, coupling=STRESS_COUPLING):
    """NV-frame stress tensor (GPa) → Barson shifts ``(Mz, Mx, My)`` in MHz.

    ``stress`` is a Voigt 6-vector ``[σxx, σyy, σzz, σyz, σxz, σxy]`` or a 3×3
    matrix, already expressed in the NV coordinate frame.  Expressions are the
    Barson combinations as implemented in ``AnmayG/peak_finder``::

        Mz = (a1 − a2)(σxx + σyy) + (a1 + 2·a2)·σzz
        Mx = −(b + c)(σyy − σxx) + (√2·b − c/√2)·2·σxz
        My = −(b + c)·2·σxy      + (√2·b − c/√2)·2·σyz

    Hydrostatic stress (σxx=σyy=σzz=P) gives ``Mz = 3·a₁·P`` and ``Mx = My = 0``.
    """
    a1, a2, b, c = coupling["a1"], coupling["a2"], coupling["b"], coupling["c"]
    xx, yy, zz, yz, xz, xy = _stress_components(stress)
    off = np.sqrt(2.0) * b - c / np.sqrt(2.0)
    Mz = (a1 - a2) * (xx + yy) + (a1 + 2.0 * a2) * zz
    Mx = -(b + c) * (yy - xx) + off * 2.0 * xz
    My = -(b + c) * 2.0 * xy + off * 2.0 * yz
    return Mz, Mx, My


def effective_DE(D, E, stress, coupling=STRESS_COUPLING):
    """``(D_eff, E_eff)`` that an ODMR fit would report for an NV under ``stress``.

    ``D_eff = D + Mz``;  ``E_eff = |E·x̂ − Mx·x̂ + My·ŷ|`` i.e. the magnitude of
    the combined transverse term, ``hypot(E − Mx, My)`` (the fitted ``E`` is
    non-negative and its phase is not observable in the splitting alone).
    """
    Mz, Mx, My = stress_to_shifts(stress, coupling)
    return D + Mz, float(np.hypot(E - Mx, My))


def hydrostatic_D_shift(P_GPa):
    """ΔD (MHz) from hydrostatic pressure ``P_GPa`` — ``3·a₁·P``."""
    return DDP_HYDROSTATIC_MHZ_PER_GPA * np.asarray(P_GPa, dtype=float)


def pressure_from_D_shift(dD_MHz):
    """Hydrostatic pressure (GPa) implied by a ZFS shift ``dD_MHz`` (= D_eff − D₀).

    Assumes a purely hydrostatic stress state; any measured ``E`` indicates a
    non-hydrostatic component this scalar inversion does not capture.
    """
    return np.asarray(dD_MHz, dtype=float) / DDP_HYDROSTATIC_MHZ_PER_GPA


def sigma_from_D(D_MHz, D0=D_GS_MHZ, model="uniaxial_100"):
    """Equivalent scalar stress (GPa, ``+`` = compression) from a measured ``D``.

    On a [100]-cut diamond *any* diagonal crystal-frame stress shifts ``D`` by
    ``a₁·tr(σ) = 4.86·tr(σ)`` MHz/GPa — identically for every NV family — so
    ``D`` alone constrains only ``tr(σ)``.  ``model`` maps the trace to a
    reported scalar:

    * ``"uniaxial_100"`` : σ along one ⟨100⟩ axis, ``tr = σ``   → 4.86 MHz/GPa
    * ``"biaxial"``      : σxx = σyy, σzz = 0, ``tr = 2σ``       → 9.72 MHz/GPa
    * ``"hydrostatic"``  : σxx = σyy = σzz = P, ``tr = 3P``      → 14.58 MHz/GPa

    Vectorised — pass a per-pixel ``D`` map.  ``D0`` is the unstressed reference
    (nominal 2870.0; real diamonds scatter ±0.3 MHz → ±0.06 GPa floor).  Note a
    fitted line centre ``(f01+f02)/2`` also carries a 2nd-order B⊥ offset;
    subtract a model centre (:func:`nv_lines`) first if the absolute level
    matters.  ``E`` is the check on the assumption — see :func:`stress_from_DE`.
    """
    a1 = STRESS_COUPLING["a1"]
    factor = {"uniaxial_100": a1, "biaxial": 2.0 * a1, "hydrostatic": 3.0 * a1}
    if model not in factor:
        raise ValueError(f"model must be one of {sorted(factor)}, got {model!r}")
    return (np.asarray(D_MHz, dtype=float) - D0) / factor[model]


def stress_from_DE(D_eff, E, D0=D_GS_MHZ, model="hydrostatic",
                   coupling=STRESS_COUPLING):
    """Back out a scalar stress (GPa) from a fitted ``(D_eff, E)`` under an
    *assumed* stress state.

    A full 6-component tensor cannot be recovered from one symmetric NV-family
    pair (all this [100]-cut data resolves).  peak_finder gets the tensor only
    by fitting all four families simultaneously.  Here you pick the state:

    ``model="hydrostatic"`` : σxx=σyy=σzz=σ.  ``σ = (D_eff − D0)/(3·a₁)``;
        ``E`` should be ~0 and is returned as ``E_residual`` (the
        non-hydrostatic transverse shift).
    ``model="biaxial"`` : in-plane σxx=σyy=σ, σzz=0 (thin film on a substrate).
        ``σ = (D_eff − D0)/(2·(a₁ − a₂))``; ``E`` again purely non-biaxial.
    ``model="uniaxial_z"`` : σzz=σ only.  ``σ = (D_eff − D0)/(a₁ + 2·a₂)``.

    Returns a dict ``{"stress_GPa", "E_residual_MHz", "model"}``.
    """
    a1, a2 = coupling["a1"], coupling["a2"]
    dD = float(D_eff) - float(D0)
    denom = {
        "hydrostatic": 3.0 * a1,
        "biaxial": 2.0 * (a1 - a2),
        "uniaxial_z": a1 + 2.0 * a2,
    }
    if model not in denom:
        raise ValueError(f"model must be one of {sorted(denom)}, got {model!r}")
    return {
        "stress_GPa": dD / denom[model],
        "E_residual_MHz": float(E),
        "model": model,
    }


# --------------------------------------------------------------------------- #
# Thermal shift  D(T)
# --------------------------------------------------------------------------- #
# Cambria et al. 2023 two-phonon-mode model: D(T) = D0 + Σ cᵢ nᵢ(T).
_CAMBRIA2023 = {
    "D0": 2877.38,       # MHz, ZFS at 0 K
    "c": (-54.91, -249.6),        # MHz
    "delta_meV": (58.73, 145.5),  # phonon mode energies
}
# Chen et al. 2011 5th-order polynomial (coefficients dᵢ for Σ dᵢ Tⁱ, MHz/Kⁱ),
# as reproduced in Cambria 2023 Table S1.
_CHEN2011 = (2877.71, -4.625e-3, 1.067e-4, -9.325e-7, 1.739e-9, -1.838e-12)


def _bose(delta_meV, T_K):
    T = np.asarray(T_K, dtype=float)
    with np.errstate(over="ignore", divide="ignore"):
        return 1.0 / np.expm1(delta_meV / (_KB_MEV_PER_K * T))


def D_of_T(T_K, model="cambria2023"):
    """NV zero-field splitting ``D`` (MHz) at temperature ``T_K``.

    ``model="cambria2023"`` (default): two-phonon-mode occupation model, valid
    15–500 K, does not diverge outside that range.  ``model="chen2011"``:
    5th-order polynomial, best 0–300 K.  The two differ by a roughly constant
    ~330 kHz sample offset (Cambria's diamond sits low); use ``chen2011`` if you
    want agreement with the more commonly cited absolute values.
    """
    if model == "cambria2023":
        p = _CAMBRIA2023
        D = p["D0"]
        for ci, di in zip(p["c"], p["delta_meV"]):
            D = D + ci * _bose(di, T_K)
        return D
    if model == "chen2011":
        return np.polyval(_CHEN2011[::-1], np.asarray(T_K, dtype=float))
    raise ValueError(f"model must be 'cambria2023' or 'chen2011', got {model!r}")


def dDdT_of_T(T_K, model="cambria2023"):
    """Analytic dD/dT (MHz/K) at ``T_K`` for the chosen :func:`D_of_T` model.

    Near 300 K this is ≈ −0.07 MHz/K; it flattens toward 0 below ~50 K.
    """
    T = np.asarray(T_K, dtype=float)
    if model == "cambria2023":
        p = _CAMBRIA2023
        out = np.zeros_like(T)
        for ci, di in zip(p["c"], p["delta_meV"]):
            x = di / (_KB_MEV_PER_K * T)
            ex = np.exp(x)
            # d/dT [1/(e^x - 1)] with x = Δ/(kB T):  (Δ/(kB T²))·e^x/(e^x−1)²
            out = out + ci * (di / (_KB_MEV_PER_K * T**2)) * ex / (ex - 1.0) ** 2
        return out
    if model == "chen2011":
        deriv = np.polynomial.Polynomial(_CHEN2011).deriv()
        return deriv(T)
    raise ValueError(f"model must be 'cambria2023' or 'chen2011', got {model!r}")


# --------------------------------------------------------------------------- #
# Stark (electric-field) shift
# --------------------------------------------------------------------------- #
def stark_shifts(Epar_V_per_cm=0.0, Eperp_V_per_cm=0.0):
    """Electric-field shifts ``(dD, dE)`` in MHz for a field in V/cm.

    ``dD = d_∥·E_∥`` and ``dE = d_⊥·E_⊥`` with ``d_∥ = 0.35``, ``d_⊥ = 17``
    Hz·cm/V (van Oort & Glasbeek 1990).  ``dE`` adds to the ``E`` parameter.
    """
    dD = STARK_PAR_HZ_CM_PER_V * np.asarray(Epar_V_per_cm, dtype=float) / 1e6
    dE = STARK_PERP_HZ_CM_PER_V * np.asarray(Eperp_V_per_cm, dtype=float) / 1e6
    return dD, dE
