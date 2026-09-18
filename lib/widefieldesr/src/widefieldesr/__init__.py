"""widefieldesr — widefield NV-ESR magnetometry imaging analysis.

Pipeline: load ``.mat`` acquisitions, compute per-pixel ESR contrast, select
ROIs, fit N-Lorentzian dips per pixel (CPU pool or batched CUDA), plot, and
persist analysis metadata / fit maps.

Data and sidecar locations are configured via environment variables — see
:mod:`widefieldesr.config`.

Convention in notebooks: ``import widefieldesr as wh``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from labplot import (
    my_plot_frame,
    cbar_postprocess,
    coolwarm_white,
    coolwarm_black,
    coolwarm_gray,
)

from .config import data_root, analysis_root, output_root, relative_to_data
from .io import (
    load_esr,
    load_field_txt,
    metadata_dir,
    output_dir,
    load_metadata,
    save_metadata,
    save_fit_maps,
    load_fit_maps,
    export_session_plots,
)
from .preprocess import compute_contrast, dip_depth_map, spatial_avg_spectrum
from .binning import bin_mask, bin_contrast, upsample_maps, smooth_contrast
from .roi import (
    make_rect_roi,
    make_circle_roi,
    make_ellipse_roi,
    roi_spectra,
    roi_avg_spectrum,
)
from .fitting import (
    init_from_spectrum,
    eval_model,
    fit_esr_spectrum,
    fit_roi_cpu,
    fit_roi_gpu,
    fit_roi,
)
from .nv_physics import (
    GAMMA_MHZ_PER_G,
    NV_PROJECTION,
    D_GS_MHZ,
    STRESS_COUPLING,
    DDP_HYDROSTATIC_MHZ_PER_GPA,
    nv_hamiltonian,
    nv_lines,
    splitting_of_B100,
    make_B100_inverter,
    stress_to_shifts,
    effective_DE,
    hydrostatic_D_shift,
    pressure_from_D_shift,
    sigma_from_D,
    stress_from_DE,
    D_of_T,
    dDdT_of_T,
    stark_shifts,
)
from .plotting import (
    plot_image,
    plot_image_with_roi,
    plot_dual_spectrum,
    plot_spectrum_with_fit,
)
from .current_reconstruction import reconstruct_current

__all__ = [
    # passthrough plotting style
    "plt",
    "my_plot_frame",
    "cbar_postprocess",
    "coolwarm_white",
    "coolwarm_black",
    "coolwarm_gray",
    # config
    "data_root",
    "analysis_root",
    "output_root",
    "relative_to_data",
    # io
    "load_esr",
    "load_field_txt",
    "metadata_dir",
    "output_dir",
    "load_metadata",
    "save_metadata",
    "save_fit_maps",
    "load_fit_maps",
    "export_session_plots",
    # preprocess
    "compute_contrast",
    "dip_depth_map",
    "spatial_avg_spectrum",
    # binning
    "bin_mask",
    "bin_contrast",
    "upsample_maps",
    "smooth_contrast",
    # roi
    "make_rect_roi",
    "make_circle_roi",
    "make_ellipse_roi",
    "roi_spectra",
    "roi_avg_spectrum",
    # fitting
    "init_from_spectrum",
    "eval_model",
    "fit_esr_spectrum",
    "fit_roi_cpu",
    "fit_roi_gpu",
    "fit_roi",
    # nv_physics
    "GAMMA_MHZ_PER_G",
    "NV_PROJECTION",
    "D_GS_MHZ",
    "STRESS_COUPLING",
    "DDP_HYDROSTATIC_MHZ_PER_GPA",
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
    # plotting
    "plot_image",
    "plot_image_with_roi",
    "plot_dual_spectrum",
    "plot_spectrum_with_fit",
    # current_reconstruction
    "reconstruct_current",
]
