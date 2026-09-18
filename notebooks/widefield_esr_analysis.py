import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    import widefieldesr as wh

    return mo, wh


@app.cell
def _(mo, wh):
    file_browser = mo.ui.file_browser(
        initial_path=wh.data_root(),
        filetypes=[".mat"],
        selection_mode="file",
        multiple=False,
        label="ESR cube",
    )
    file_browser
    return (file_browser,)


@app.cell
def _(file_browser, mo, wh):
    # Falls back to the bundled sample until something is picked above.
    _default = wh.data_root() / "sample_odmr.mat"
    data_path = str(file_browser.path(0)) if file_browser.value else str(_default)
    _data = wh.load_esr(data_path)
    signal = _data["signal"]       # (ny, nx, n_freq)
    freq_GHz = _data["freq_GHz"]   # (n_freq,) in GHz
    n_freq = len(freq_GHz)         # NOTE: was `_data["N"]` (avg count, wrong
                                    # concept -- caused an IndexError on any
                                    # dataset where N >= len(freq_GHz), which
                                    # this example's small sample tripped).

    mo.md(f"Loaded: `{data_path}`")
    return data_path, freq_GHz, n_freq, signal


@app.cell
def _(signal, wh):
    contrast = wh.compute_contrast(signal)
    avg_spectrum = wh.spatial_avg_spectrum(signal)
    avg_contrast = wh.spatial_avg_spectrum(contrast)
    dip_map = wh.dip_depth_map(contrast)
    return avg_contrast, avg_spectrum, contrast, dip_map


@app.cell
def _(data_path, wh):
    _meta_init = wh.load_metadata(data_path)
    last_roi_init = _meta_init.get("last_roi", {})
    return (last_roi_init,)


@app.cell
def _(data_path, mo, wh):
    get_meta, set_meta = mo.state(wh.load_metadata(data_path))
    return get_meta, set_meta


@app.cell
def _(get_meta):
    session_meta = get_meta()
    return (session_meta,)


@app.cell
def _(mo, session_meta):
    _pixels = session_meta.get("pixels", {})
    _rows = [{"Name": k, "x": v["x"], "y": v["y"]} for k, v in _pixels.items()]
    px_table = mo.ui.table(_rows, selection="multi") if _rows else None
    return (px_table,)


@app.cell
def _(mo, px_table):
    _sel = px_table.value[0] if (px_table is not None and px_table.value) else None
    px_name_input = mo.ui.text(value=_sel["Name"] if _sel else "", label="Name")
    px_x_input = mo.ui.number(0, 511, value=int(_sel["x"]) if _sel else 256, step=1, label="x")
    px_y_input = mo.ui.number(0, 511, value=int(_sel["y"]) if _sel else 256, step=1, label="y")
    add_px_btn = mo.ui.run_button(label="Save")
    delete_px_btn = mo.ui.run_button(label="Delete")
    return add_px_btn, delete_px_btn, px_name_input, px_x_input, px_y_input


@app.cell
def _(
    add_px_btn,
    delete_px_btn,
    mo,
    px_name_input,
    px_table,
    px_x_input,
    px_y_input,
):
    mo.vstack([
        mo.md("## Pixel Manager"),
        px_table if px_table is not None else mo.md("_No saved pixels yet — add one below_"),
        mo.hstack([px_name_input, px_x_input, px_y_input, add_px_btn, delete_px_btn]),
    ])
    return


@app.cell
def _(
    add_px_btn,
    data_path,
    mo,
    px_name_input,
    px_x_input,
    px_y_input,
    session_meta,
    set_meta,
    wh,
):
    mo.stop(not add_px_btn.value)
    _updated = {**session_meta, "pixels": {**session_meta.get("pixels", {})}}
    _updated["pixels"][px_name_input.value] = {"x": int(px_x_input.value), "y": int(px_y_input.value)}
    wh.save_metadata(data_path, _updated)
    set_meta(_updated)
    return


@app.cell
def _(data_path, delete_px_btn, mo, px_table, session_meta, set_meta, wh):
    mo.stop(not delete_px_btn.value or px_table is None or not px_table.value)
    _updated = {**session_meta, "pixels": {
        k: v for k, v in session_meta.get("pixels", {}).items()
        if k not in {r["Name"] for r in px_table.value}
    }}
    wh.save_metadata(data_path, _updated)
    set_meta(_updated)
    return


@app.cell
def _(px_table):
    selected_pixels = [{"name": r["Name"], "x": r["x"], "y": r["y"]}
                       for r in (px_table.value if px_table is not None else [])]
    pixel_x = int(selected_pixels[0]["x"]) if selected_pixels else 256
    pixel_y = int(selected_pixels[0]["y"]) if selected_pixels else 256
    return (selected_pixels,)


@app.cell
def _(mo, n_freq):
    freq_slider = mo.ui.slider(0, n_freq - 1, value=n_freq // 2, label="Frequency index")
    return (freq_slider,)


@app.cell
def _(freq_slider):
    freq_idx = freq_slider.value
    return (freq_idx,)


@app.cell
def _(freq_GHz, freq_idx, freq_slider, mo):
    mo.vstack([
        mo.md("## Widefield ESR — Bi-2212-S1"),
        mo.hstack([freq_slider, mo.md(f"**{freq_GHz[freq_idx]:.4f} GHz**")]),
    ])
    return


@app.cell(hide_code=True)
def _(contrast, freq_GHz, freq_idx, mo, selected_pixels, signal, wh):
    _COLORS = ["#ffffff", "#00ffff", "#00ff00", "#ffff00", "#ff69b4", "#ffa500"]
    _fig_pl = wh.plot_image(
        signal[:, :, freq_idx],
        title=f"PL  @  {freq_GHz[freq_idx]:.4f} GHz", cmap="hot",
    )
    _ax_pl = wh.plt.gca()
    for _i, _px in enumerate(selected_pixels):
        _ax_pl.axvline(_px["x"], color=_COLORS[_i % 6], lw=0.8, ls=":")
        _ax_pl.axhline(_px["y"], color=_COLORS[_i % 6], lw=0.8, ls=":")

    _fig_ct = wh.plot_image(
        contrast[:, :, freq_idx],
        title=f"Contrast  @  {freq_GHz[freq_idx]:.4f} GHz",
        cmap=wh.coolwarm_white, center_zero=True,
    )
    _ax_ct = wh.plt.gca()
    for _i, _px in enumerate(selected_pixels):
        _ax_ct.axvline(_px["x"], color=_COLORS[_i % 6], lw=0.8, ls=":")
        _ax_ct.axhline(_px["y"], color=_COLORS[_i % 6], lw=0.8, ls=":")

    mo.hstack([_fig_pl, _fig_ct])
    return


@app.cell
def _(dip_map, selected_pixels, wh):
    _COLORS = ["#ffffff", "#00ffff", "#00ff00", "#ffff00", "#ff69b4", "#ffa500"]
    _fig = wh.plot_image(dip_map, title="ESR dip depth map  (min contrast dF/F)", cmap=wh.coolwarm_white)
    _ax_ct = wh.plt.gca()
    for _i, _px in enumerate(selected_pixels):
        _ax_ct.axvline(_px["x"], color=_COLORS[_i % 6], lw=0.8, ls=":")
        _ax_ct.axhline(_px["y"], color=_COLORS[_i % 6], lw=0.8, ls=":")
    _fig
    return


@app.cell
def _(contrast, freq_GHz, freq_idx, mo, selected_pixels, signal, wh):
    _COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    _fig, _axs = wh.my_plot_frame(90, 45, xlabel="Frequency (GHz)", ylabel="PL (counts)", ysubplots=2)
    _ax_pl = _axs[0, 0]
    _ax_ct = _axs[1, 0]
    _ax_ct.set_ylabel("Contrast dF/F", fontsize=7)
    _ax_pl.set_title("Selected pixel spectra", fontsize=7)
    for _i, _px in enumerate(selected_pixels):
        _c = _COLORS[_i % 6]
        _ax_pl.plot(freq_GHz, signal[_px["y"], _px["x"], :], color=_c, lw=1, label=_px["name"])
        _ax_ct.plot(freq_GHz, contrast[_px["y"], _px["x"], :], color=_c, lw=1)
    if selected_pixels:
        _ax_pl.legend(fontsize=6, frameon=False)
        for _ax in _axs[:, 0]:
            _ax.axvline(freq_GHz[freq_idx], color="gray", lw=0.6, ls="--", alpha=0.7)
    mo.md("_Select pixels in the Pixel Manager table above to display spectra here_") if not selected_pixels else _fig
    return


@app.cell
def _(avg_contrast, avg_spectrum, freq_GHz, freq_idx, wh):
    _fig = wh.plot_dual_spectrum(
        freq_GHz, avg_spectrum, avg_contrast,
        title="Spatial average", freq_marker=freq_GHz[freq_idx],
    )
    _fig
    return


@app.cell
def _(mo, session_meta):
    _rois = session_meta.get("rois", {})
    _roi_rows = [
        {"Name": k, "shape": v.get("shape", "?"),
         "cx": v.get("cx", 0), "cy": v.get("cy", 0),
         "a": v.get("a", 0), "b": v.get("b", 0), "n_peaks": v.get("n_peaks", 2),
         "x_bin": v.get("x_bin", 1), "y_bin": v.get("y_bin", 1),
         "bin_mode": v.get("bin_mode", "block"),
         "per_pixel_init": v.get("per_pixel_init", False),
         "init_smooth_n": v.get("init_smooth_n", 1)}
        for k, v in _rois.items()
    ]
    roi_table = mo.ui.table(_roi_rows, selection="single") if _roi_rows else None
    return (roi_table,)


@app.cell
def _(last_roi_init, mo, roi_table, session_meta):
    _sel = roi_table.value[0] if (roi_table is not None and roi_table.value) else None
    _r = _sel if _sel else last_roi_init
    _name_def = _sel["Name"] if _sel else "roi_1"
    _px_names = list(session_meta.get("pixels", {}).keys())
    roi_lib_name = mo.ui.text(value=_name_def, label="Name")
    roi_shape_input = mo.ui.dropdown(["Rect", "Circle", "Ellipse"],
                                      value=_r.get("shape", "Rect"), label="Shape")
    roi_px_center = mo.ui.dropdown(["— manual —"] + _px_names, value="— manual —",
                                    label="Center from pixel")
    roi_cx_input = mo.ui.number(0, 511, value=int(_r.get("cx", 256)), step=1, label="cx")
    roi_cy_input = mo.ui.number(0, 511, value=int(_r.get("cy", 256)), step=1, label="cy")
    roi_a_input = mo.ui.number(5, 300, value=int(_r.get("a", 100)), step=1,
                                label="a  (half-width / radius)")
    roi_b_input = mo.ui.number(5, 300, value=int(_r.get("b", 80)), step=1,
                                label="b  (half-height, rect/ellipse)")
    n_peaks_ui = mo.ui.number(1, 4, value=int(_r.get("n_peaks", 2)), step=1, label="N peaks")
    roi_xbin_input = mo.ui.number(1, 64, value=int(_r.get("x_bin", 1)), step=1, label="x bin")
    roi_ybin_input = mo.ui.number(1, 64, value=int(_r.get("y_bin", 1)), step=1, label="y bin")
    roi_binmode_input = mo.ui.dropdown(
        ["block", "sliding"], value=_r.get("bin_mode", "block"), label="bin mode",
    )
    per_pixel_init_ui = mo.ui.checkbox(
        value=bool(_r.get("per_pixel_init", False)),
        label="Per-pixel init (seed each pixel from its own spectrum, not the shared one)",
    )
    init_smooth_ui = mo.ui.number(
        1, 31, value=int(_r.get("init_smooth_n", 1)), step=1,
        label="init smooth (pts, peak-search only)",
    )
    run_button = mo.ui.run_button(label="Run pixel-wise fit")
    add_roi_btn = mo.ui.run_button(label="Save")
    delete_roi_btn = mo.ui.run_button(label="Delete selected")
    return (
        add_roi_btn,
        delete_roi_btn,
        init_smooth_ui,
        n_peaks_ui,
        per_pixel_init_ui,
        roi_a_input,
        roi_b_input,
        roi_binmode_input,
        roi_cx_input,
        roi_cy_input,
        roi_lib_name,
        roi_px_center,
        roi_shape_input,
        roi_xbin_input,
        roi_ybin_input,
        run_button,
    )


@app.cell
def _(
    add_roi_btn,
    delete_roi_btn,
    init_smooth_ui,
    mo,
    n_peaks_ui,
    per_pixel_init_ui,
    roi_a_input,
    roi_b_input,
    roi_binmode_input,
    roi_cx_input,
    roi_cy_input,
    roi_lib_name,
    roi_px_center,
    roi_shape_input,
    roi_table,
    roi_xbin_input,
    roi_ybin_input,
    run_button,
):
    mo.vstack([
        mo.md("## ROI Manager"),
        roi_table if roi_table is not None else mo.md("_No saved ROIs yet — define one below_"),
        mo.md("---"),
        mo.hstack([roi_lib_name, roi_shape_input, n_peaks_ui, run_button]),
        mo.hstack([roi_px_center, roi_cx_input, roi_cy_input]),
        mo.hstack([roi_a_input, roi_b_input, roi_xbin_input, roi_ybin_input, roi_binmode_input]),
        mo.md(
            "_Bin mode **block** (default): non-overlapping x/y-bin blocks, one "
            "fit per block, replicated back onto every raw pixel it covers — "
            "fewer/faster fits, blocky output. **sliding**: a moving x/y-bin "
            "window average centred on every pixel — every pixel is still fit "
            "individually (no speedup) but from a smoothed, higher-SNR "
            "spectrum, with no blockiness. x/y bin = 1 is a no-op either way._"
        ) if roi_xbin_input.value > 1 or roi_ybin_input.value > 1 else mo.md(""),
        mo.hstack([per_pixel_init_ui, init_smooth_ui]),
        mo.md(
            "_Per-pixel init ignores the init sliders below and seeds each pixel "
            "from its own spectrum's deepest minima instead of the shared "
            "ROI-average seed. This can help when the dip position varies a lot "
            "across the ROI, but an individual pixel's spectrum is noisier than "
            "the ROI average — its own peak search can lock onto a spurious "
            "minimum and converge to the wrong dip pair. **init smooth** "
            "smooths a copy of each spectrum before that peak search only (not "
            "the data actually fit) and is the natural fix — try ~9-15 pts. "
            "Compare against the default (unchecked) result before trusting it._"
        ) if per_pixel_init_ui.value else mo.md(""),
        mo.hstack([add_roi_btn, delete_roi_btn]),
    ])
    return


@app.cell
def _(roi_table, session_meta):
    _rois = session_meta.get("rois", {})
    active_roi = None
    if roi_table is not None and roi_table.value:
        _sel_name = roi_table.value[0]["Name"]
        active_roi = _rois.get(_sel_name)
    return (active_roi,)


@app.cell
def _(
    add_roi_btn,
    custom_init,
    data_path,
    fit_range_ui,
    init_smooth_ui,
    linear_bg_ui,
    mo,
    n_peaks_ui,
    per_pixel_init_ui,
    roi_a_input,
    roi_b_input,
    roi_binmode_input,
    roi_cx_input,
    roi_cy_input,
    roi_lib_name,
    roi_px_center,
    roi_shape_input,
    roi_xbin_input,
    roi_ybin_input,
    session_meta,
    set_meta,
    wh,
):
    mo.stop(not add_roi_btn.value)
    _cx = roi_cx_input.value
    _cy = roi_cy_input.value
    if roi_px_center.value != "— manual —":
        _pxdata = session_meta["pixels"][roi_px_center.value]
        _cx, _cy = _pxdata["x"], _pxdata["y"]
    _roi_config = {
        "shape": roi_shape_input.value,
        "cx": int(_cx), "cy": int(_cy),
        "a": int(roi_a_input.value), "b": int(roi_b_input.value),
        "n_peaks": int(n_peaks_ui.value),
        "x_bin": int(roi_xbin_input.value), "y_bin": int(roi_ybin_input.value),
        "bin_mode": roi_binmode_input.value,
        "per_pixel_init": bool(per_pixel_init_ui.value),
        "init_smooth_n": int(init_smooth_ui.value),
        "init_params": custom_init,
        "fit_range": list(fit_range_ui.value),
        "linear_bg": bool(linear_bg_ui.value),
    }
    _updated = {**session_meta, "rois": {**session_meta.get("rois", {})}}
    _updated["rois"][roi_lib_name.value] = _roi_config
    _updated["last_roi"] = _roi_config
    wh.save_metadata(data_path, _updated)
    set_meta(_updated)
    return


@app.cell
def _(data_path, delete_roi_btn, mo, roi_table, session_meta, set_meta, wh):
    mo.stop(not delete_roi_btn.value or roi_table is None or not roi_table.value)
    _del_name = roi_table.value[0]["Name"]
    _updated = {**session_meta, "rois": {
        k: v for k, v in session_meta.get("rois", {}).items() if k != _del_name
    }}
    wh.save_metadata(data_path, _updated)
    set_meta(_updated)
    return


@app.cell
def _(mo):
    get_refresh, set_refresh = mo.state(0)
    refresh = get_refresh()
    return (set_refresh,)


@app.cell
def _(data_path, mo, wh):
    _meta_fm = wh.load_metadata(data_path)
    _fit_rows = [
        {
            "ROI": _rname, "Fit": _fname, "n_peaks": _fdata.get("n_peaks", "?"),
            "range": (
                f"{_fdata['fit_range'][0]:.3f}–{_fdata['fit_range'][1]:.3f} GHz"
                if _fdata.get("fit_range") else "full"
            ),
            "bg": "linear" if _fdata.get("linear_bg") else "const",
        }
        for _rname, _rdata in _meta_fm.get("rois", {}).items()
        for _fname, _fdata in _rdata.get("fits", {}).items()
    ]
    fit_table = mo.ui.table(_fit_rows, selection="single") if _fit_rows else None
    delete_fit_btn = mo.ui.run_button(label="Delete selected fit")
    return delete_fit_btn, fit_table


@app.cell
def _(delete_fit_btn, fit_table, mo):
    mo.vstack([
        mo.md("### Fitting Manager"),
        fit_table if fit_table is not None else mo.md("_No saved fits yet_"),
        delete_fit_btn,
    ])
    return


@app.cell
def _(data_path, delete_fit_btn, fit_table, mo, set_meta, wh):
    mo.stop(not delete_fit_btn.value or fit_table is None or not fit_table.value)
    _roi_name = fit_table.value[0]["ROI"]
    _fit_name = fit_table.value[0]["Fit"]
    _disk = wh.load_metadata(data_path)
    _disk["rois"][_roi_name]["fits"] = {
        k: v for k, v in _disk["rois"].get(_roi_name, {}).get("fits", {}).items()
        if k != _fit_name
    }
    wh.save_metadata(data_path, _disk)
    set_meta(_disk)
    return


@app.cell
def _(
    active_roi,
    roi_a_input,
    roi_b_input,
    roi_cx_input,
    roi_cy_input,
    roi_shape_input,
    signal,
    wh,
):
    # img_shape passed explicitly (rather than relying on wh.make_*_roi's
    # own default of (512,512), the real camera's native FOV) so this
    # works against any image size, e.g. this example's small sample.
    _img_shape = signal.shape[:2]
    if active_roi is not None:
        _shape = active_roi.get("shape", "Rect")
        _cx, _cy = int(active_roi["cx"]), int(active_roi["cy"])
        _a = int(active_roi["a"])
        _b = int(active_roi.get("b", active_roi["a"]))
    else:
        _shape = roi_shape_input.value
        _cx, _cy = int(roi_cx_input.value), int(roi_cy_input.value)
        _a, _b = int(roi_a_input.value), int(roi_b_input.value)
    if _shape == "Rect":
        roi_mask = wh.make_rect_roi(int(_cx - _a), int(_cy - _b), 2 * _a, 2 * _b, img_shape=_img_shape)
        roi_overlay = ("rect", (_cx, _cy, _a, _b))
    elif _shape == "Circle":
        roi_mask = wh.make_circle_roi(_cx, _cy, _a, img_shape=_img_shape)
        roi_overlay = ("circle", (_cx, _cy, _a))
    else:
        roi_mask = wh.make_ellipse_roi(_cx, _cy, _a, _b, img_shape=_img_shape)
        roi_overlay = ("ellipse", (_cx, _cy, _a, _b))
    return roi_mask, roi_overlay


@app.cell
def _(freq_GHz, freq_idx, mo, roi_overlay, selected_pixels, signal, wh):
    _rtype, _rparams = roi_overlay
    _fig = wh.plot_image_with_roi(
        signal[:, :, freq_idx],
        roi_type=_rtype, roi_params=_rparams, pixels=selected_pixels,
        title=f"PL @ {freq_GHz[freq_idx]:.4f} GHz  —  ROI (cyan)",
        cmap="hot",
    )
    mo.md("") if False else _fig
    return


@app.cell
def _(contrast, freq_GHz, n_peaks_ui, roi_mask, wh):
    roi_avg_contrast = wh.roi_avg_spectrum(contrast, roi_mask)
    # Quick-look orientation fit only (constant offset). The linear-background
    # option lives in the Init Parameter Tuning section and applies to the
    # tuning preview, the per-pixel fit, and the export.
    roi_fit = wh.fit_esr_spectrum(freq_GHz, roi_avg_contrast, n_peaks=n_peaks_ui.value)
    return roi_avg_contrast, roi_fit


@app.cell
def _(freq_GHz, n_peaks_ui, roi_avg_contrast, roi_fit, wh):
    _fig = wh.plot_spectrum_with_fit(
        freq_GHz, roi_avg_contrast, fit_result=roi_fit,
        title=f"ROI average contrast  —  {n_peaks_ui.value}-peak fit",
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ### Init Parameter Tuning
    """)
    return


@app.cell
def _(active_roi, freq_GHz, last_roi_init, mo):
    # Restrict fitting to part of the spectrum. Default = full range, so ROIs /
    # fits saved before this control existed (no "fit_range" key) load unchanged.
    _r = active_roi if active_roi is not None else last_roi_init
    _lo, _hi = float(freq_GHz[0]), float(freq_GHz[-1])
    _saved = (_r or {}).get("fit_range")
    _v0, _v1 = (float(_saved[0]), float(_saved[1])) if _saved else (_lo, _hi)
    fit_range_ui = mo.ui.range_slider(
        start=_lo, stop=_hi, step=0.001, value=[_v0, _v1],
        show_value=True, full_width=True, label="Fit range (GHz)",
    )
    fit_range_ui
    return (fit_range_ui,)


@app.cell
def _(active_roi, last_roi_init, mo):
    # Opt-in linear background (offset + slope·f). Default off, so ROIs / fits
    # saved before this key existed load and re-fit identically.
    _r = active_roi if active_roi is not None else last_roi_init
    linear_bg_ui = mo.ui.checkbox(
        value=bool((_r or {}).get("linear_bg", False)),
        label="Linear background (slope·f)",
    )
    linear_bg_ui
    return (linear_bg_ui,)


@app.cell
def _(fit_range_ui, freq_GHz, mo):
    fit_mask = (freq_GHz >= fit_range_ui.value[0]) & (freq_GHz <= fit_range_ui.value[1])
    mo.md(
        "_full spectrum_" if bool(fit_mask.all())
        else f"_fitting {int(fit_mask.sum())} / {len(freq_GHz)} points "
             f"({fit_range_ui.value[0]:.4f}–{fit_range_ui.value[1]:.4f} GHz)_"
    )
    return (fit_mask,)


@app.cell
def _(mo):
    get_loaded_p, set_loaded_p = mo.state({})
    return get_loaded_p, set_loaded_p


@app.cell
def _(get_loaded_p):
    loaded_p = get_loaded_p()
    return (loaded_p,)


@app.cell
def _(data_path, mo, n_peaks_ui, wh):
    _meta_ls = wh.load_metadata(data_path)
    _compat = [
        f"{_rname} / {_fname}"
        for _rname, _rdata in _meta_ls.get("rois", {}).items()
        for _fname, _fdata in _rdata.get("fits", {}).items()
        if _fdata.get("n_peaks") == n_peaks_ui.value
    ]
    init_load_src = mo.ui.dropdown(
        ["— auto —"] + _compat, value="— auto —", label="Load init from saved fit"
    )
    load_init_btn = mo.ui.run_button(label="Load")
    mo.hstack([init_load_src, load_init_btn])
    return init_load_src, load_init_btn


@app.cell
def _(init_load_src, load_init_btn, loaded_p, mo):
    mo.vstack([
              mo.md(f"dropdown: `{init_load_src.value}`"),
              mo.md(f"btn clicked: `{load_init_btn.value}`"),
              mo.md(f"loaded_p: `{loaded_p}`"),
          ])
    return


@app.cell
def _(data_path, init_load_src, load_init_btn, mo, set_loaded_p, wh):
    mo.stop(not load_init_btn.value or init_load_src.value == "— auto —")
    _parts = init_load_src.value.split(" / ", 1)
    _roi_name_la, _fit_name_la = _parts[0], _parts[1]
    _meta_la = wh.load_metadata(data_path)
    _saved = _meta_la.get("rois", {}).get(_roi_name_la, {}).get("fits", {}).get(_fit_name_la, {}).get("init_params", {})
    set_loaded_p(_saved)
    return


@app.cell
def _(
    active_roi,
    freq_GHz,
    loaded_p,
    mo,
    n_peaks_ui,
    roi_avg_contrast,
    selected_pixels,
    wh,
):
    _n = n_peaks_ui.value
    _p0 = wh.init_from_spectrum(freq_GHz, roi_avg_contrast, _n)
    if loaded_p:
        _p0 = {**_p0, **loaded_p}
    elif active_roi is not None and active_roi.get("n_peaks") == _n:
        _saved = active_roi.get("init_params", {})
        if _saved:
            _p0 = {**_p0, **_saved}
    _source_options = ["ROI average"] + [px["name"] for px in selected_pixels]

    init_source = mo.ui.dropdown(_source_options, value="ROI average", label="Source")
    init_offset = mo.ui.slider(-0.02, 0.02, value=round(_p0["offset"], 5), step=0.0005,
                               label=f"offset [{_p0['offset']:+.5f}]")
    init_f0 = mo.ui.array([
        mo.ui.slider(float(freq_GHz[0]), float(freq_GHz[-1]),
                     value=round(_p0[f"f0{i+1}"], 4), step=0.001,
                     label=f"f₀_{i+1} (GHz) [{_p0[f'f0{i+1}']:.4f}]")
        for i in range(_n)
    ])
    init_A = mo.ui.array([
        mo.ui.slider(0.0, 0.05, value=round(_p0[f"A{i+1}"], 5), step=0.0002,
                     label=f"A_{i+1} [{_p0[f'A{i+1}']:.5f}]")
        for i in range(_n)
    ])
    init_gamma = mo.ui.array([
        mo.ui.slider(0.001, 0.15, value=round(_p0[f"gamma{i+1}"], 4), step=0.001,
                     label=f"γ_{i+1} (GHz) [{_p0[f'gamma{i+1}']:.4f}]")
        for i in range(_n)
    ])
    fit_init_btn = mo.ui.run_button(label="Fit from init")
    return fit_init_btn, init_A, init_f0, init_gamma, init_offset, init_source


@app.cell
def _(
    contrast,
    fit_init_btn,
    fit_mask,
    freq_GHz,
    init_A,
    init_f0,
    init_gamma,
    init_offset,
    init_source,
    linear_bg_ui,
    mo,
    n_peaks_ui,
    roi_avg_contrast,
    selected_pixels,
    wh,
):
    _n = n_peaks_ui.value

    custom_init = {"offset": init_offset.value}
    for _i in range(1, _n + 1):
        custom_init[f"A{_i}"] = init_A.value[_i - 1]
        custom_init[f"f0{_i}"] = init_f0.value[_i - 1]
        custom_init[f"gamma{_i}"] = init_gamma.value[_i - 1]

    _src = init_source.value
    if _src == "ROI average" or not selected_pixels:
        _spectrum = roi_avg_contrast
        _src_label = "ROI average"
    else:
        _match = next((px for px in selected_pixels if px["name"] == _src), selected_pixels[0])
        _spectrum = contrast[_match["y"], _match["x"], :]
        _src_label = f"Pixel '{_match['name']}' ({_match['x']}, {_match['y']})"

    _model = wh.eval_model(freq_GHz, custom_init, _n)
    _fit_freq = freq_GHz[fit_mask]
    _fit_spec = _spectrum[fit_mask]
    _fit = (
        wh.fit_esr_spectrum(_fit_freq, _fit_spec, n_peaks=_n, init_params=custom_init,
                            linear_bg=linear_bg_ui.value)
        if fit_init_btn.value and _fit_freq.size >= 3 * _n + 1
        else None
    )

    _fig, _axs = wh.my_plot_frame(90, 50, xlabel="Frequency (GHz)", ylabel="Contrast dF/F")
    _ax = _axs[0, 0]
    _ax.plot(freq_GHz, _spectrum, color="#2c5f8a", lw=1, label=_src_label)
    _ax.plot(freq_GHz, _model, color="orange", lw=1, ls="--", label="init model")
    if _fit is not None:
        _ax.plot(_fit_freq, _fit.best_fit, color="r", lw=1, ls="-.", label="fit")
    if not fit_mask.all():
        _ax.axvspan(freq_GHz[0], _fit_freq[0], color="0.88", zorder=0, lw=0)
        _ax.axvspan(_fit_freq[-1], freq_GHz[-1], color="0.88", zorder=0, lw=0)
    _ax.legend(fontsize=6, frameon=False)
    _ax.set_title(f"Init tuning  —  {_n}-peak model", fontsize=7)

    _controls = mo.vstack([
        mo.hstack([init_source, fit_init_btn]),
        init_offset,
        mo.md("**f₀ centers (GHz)**"), init_f0,
        mo.md("**Amplitudes A**"), init_A,
        mo.md("**Linewidths γ (GHz)**"), init_gamma,
    ])
    mo.hstack([_controls, _fig])
    return (custom_init,)


@app.cell
def _(custom_init, fit_range_ui, n_peaks_ui):
    # f0 initial guesses clamped into the fit range, so a window narrowed
    # around one peak can't leave an init value outside the fit bounds.
    _lo, _hi = fit_range_ui.value
    fit_init = dict(custom_init)
    for _i in range(1, n_peaks_ui.value + 1):
        fit_init[f"f0{_i}"] = min(max(fit_init[f"f0{_i}"], _lo), _hi)
    return (fit_init,)


@app.cell
def _(mo):
    get_fit_maps, set_fit_maps = mo.state(None)
    return get_fit_maps, set_fit_maps


@app.cell
def _(
    contrast,
    fit_init,
    fit_mask,
    freq_GHz,
    init_smooth_ui,
    linear_bg_ui,
    mo,
    n_peaks_ui,
    per_pixel_init_ui,
    roi_binmode_input,
    roi_mask,
    roi_xbin_input,
    roi_ybin_input,
    run_button,
    set_fit_maps,
    wh,
):
    mo.stop(not run_button.value)
    set_fit_maps(wh.fit_roi(
        contrast[:, :, fit_mask], roi_mask, freq_GHz[fit_mask],
        n_peaks=n_peaks_ui.value, init_params=fit_init,
        linear_bg=linear_bg_ui.value,
        x_bin=roi_xbin_input.value, y_bin=roi_ybin_input.value,
        bin_mode=roi_binmode_input.value,
        per_pixel_init=per_pixel_init_ui.value,
        init_smooth_n=init_smooth_ui.value,
    ))
    return


@app.cell
def _(get_fit_maps):
    fit_maps = get_fit_maps()
    return (fit_maps,)


@app.cell
def _(data_path, fit_maps, fit_table, wh):
    if fit_table is not None and fit_table.value:
        _row = fit_table.value[0]
        _prefix = f"{_row['ROI']}_{_row['Fit']}"
        _loaded = wh.load_fit_maps(data_path, _prefix)
        display_maps = _loaded if _loaded is not None else fit_maps
        display_n_peaks = int(_row["n_peaks"])
    else:
        display_maps = fit_maps
        display_n_peaks = None
    return display_maps, display_n_peaks


@app.cell
def _(display_maps, mo, selected_pixels):
    mo.stop(display_maps is None)
    _px_opts = ["Manual"] + [px["name"] for px in selected_pixels]
    inspect_source = mo.ui.dropdown(_px_opts, value="Manual", label="Inspect pixel")
    inspect_x = mo.ui.slider(0, 511, value=256, step=1, label="x")
    inspect_y = mo.ui.slider(0, 511, value=256, step=1, label="y")
    return inspect_source, inspect_x, inspect_y


@app.cell
def _(inspect_source, inspect_x, inspect_y, mo, selected_pixels):
    if inspect_source.value == "Manual":
        _ix, _iy = inspect_x.value, inspect_y.value
        _loc_ctrl = mo.hstack([inspect_x, inspect_y])
    else:
        _match = next((px for px in selected_pixels if px["name"] == inspect_source.value), None)
        _ix = int(_match["x"]) if _match else 256
        _iy = int(_match["y"]) if _match else 256
        _loc_ctrl = mo.md(f"x = {_ix},  y = {_iy}")
    inspect_px = (_ix, _iy)
    mo.vstack([mo.md("#### Pixel Inspector"), inspect_source, _loc_ctrl])
    return (inspect_px,)


@app.cell
def _(display_maps, display_n_peaks, inspect_px, mo, n_peaks_ui, wh):
    mo.stop(display_maps is None)
    _n = display_n_peaks if display_n_peaks is not None else n_peaks_ui.value
    _x, _y = int(inspect_px[0]), int(inspect_px[1])
    if _n >= 2:
        _fig_map = wh.plot_image(
            display_maps["f0_2"] - display_maps["f0_1"],
            title="Splitting (GHz)", cmap=wh.coolwarm_white, center_zero=True,
        )
    else:
        _fig_map = wh.plot_image(display_maps["f0_1"], title="f₀ (GHz)", cmap="hot")
    _fig_map.axes[0].plot(_x, _y, "w+", markersize=12, markeredgewidth=1.5, clip_on=False)
    _fig_chi = wh.plot_image(display_maps["redchi"], title="Reduced χ²  (fit quality)", cmap="hot")
    _fig_chi.axes[0].plot(_x, _y, "w+", markersize=12, markeredgewidth=1.5, clip_on=False)
    mo.hstack([_fig_map, _fig_chi])
    return


@app.cell
def _(
    contrast,
    display_maps,
    display_n_peaks,
    freq_GHz,
    inspect_px,
    mo,
    n_peaks_ui,
    wh,
):
    mo.stop(display_maps is None)
    _n = display_n_peaks if display_n_peaks is not None else n_peaks_ui.value
    _x, _y = int(inspect_px[0]), int(inspect_px[1])
    _pixel_params = {"offset": float(display_maps["offset"][_y, _x])}
    if "slope" in display_maps:
        _pixel_params["slope"] = float(display_maps["slope"][_y, _x])
    for _i in range(1, _n + 1):
        _pixel_params[f"A{_i}"] = float(display_maps[f"A_{_i}"][_y, _x])
        _pixel_params[f"f0{_i}"] = float(display_maps[f"f0_{_i}"][_y, _x])
        _pixel_params[f"gamma{_i}"] = float(display_maps[f"gamma_{_i}"][_y, _x])
    _model = wh.eval_model(freq_GHz, _pixel_params, _n)
    _spectrum = contrast[_y, _x, :]
    _fig, _axs = wh.my_plot_frame(50, 30, xlabel="Frequency (GHz)", ylabel="Contrast dF/F")
    _ax = _axs[0, 0]
    _ax.plot(freq_GHz, _spectrum - _pixel_params["offset"], color="#2c5f8a", lw=1, label="data")
    _ax.plot(freq_GHz, _model - _pixel_params["offset"], color="r", lw=1, ls="--", label="fit")
    _ax.legend(fontsize=6, frameon=False)
    _fig
    return


@app.cell
def _(mo):
    export_name_input = mo.ui.text(value="fit_1", label="Export name")
    export_btn = mo.ui.run_button(label="Export fits + plots")
    return export_btn, export_name_input


@app.cell
def _(export_btn, export_name_input, fit_maps, mo, roi_table):
    _roi_name = roi_table.value[0]["Name"] if (roi_table is not None and roi_table.value) else "manual"
    _fit_name = export_name_input.value or "fit"
    _prefix = f"{_roi_name}_{_fit_name}"
    if fit_maps is not None:
        _status = f"Will save as **{_prefix}**_\\*.npz / \\*.pdf"
    else:
        _status = "Run pixel-wise fit first"
    mo.vstack([
        mo.md("### Export"),
        mo.hstack([export_name_input, export_btn]),
        mo.md(f"_{_status}_"),
    ])
    return


@app.cell
def _(
    active_roi,
    data_path,
    export_btn,
    export_name_input,
    fit_init,
    fit_maps,
    fit_mask,
    fit_range_ui,
    freq_GHz,
    init_smooth_ui,
    linear_bg_ui,
    mo,
    n_peaks_ui,
    per_pixel_init_ui,
    roi_avg_contrast,
    roi_binmode_input,
    roi_table,
    roi_xbin_input,
    roi_ybin_input,
    set_refresh,
    wh,
):
    mo.stop(not export_btn.value or fit_maps is None)
    _fit_name = export_name_input.value or "fit"
    _roi_name = roi_table.value[0]["Name"] if (roi_table is not None and roi_table.value) else "manual"
    _prefix = f"{_roi_name}_{_fit_name}"
    wh.save_fit_maps(data_path, _prefix, fit_maps)
    _splitting = fit_maps["f0_2"] - fit_maps["f0_1"] if n_peaks_ui.value >= 2 else fit_maps["f0_1"]
    _split_title = "Splitting (GHz)" if n_peaks_ui.value >= 2 else "f₀ (GHz)"
    _fit_freq = freq_GHz[fit_mask]
    _fit_spec = roi_avg_contrast[fit_mask]
    _roi_fit_export = wh.fit_esr_spectrum(_fit_freq, _fit_spec,
                                           n_peaks=n_peaks_ui.value, init_params=fit_init,
                                           linear_bg=linear_bg_ui.value)
    _best_params = {name: float(p.value) for name, p in _roi_fit_export.params.items()}
    _figs = {
        "roi_avg_fit": wh.plot_spectrum_with_fit(_fit_freq, _fit_spec, _roi_fit_export,
                                                  title=f"{_prefix}  —  ROI avg"),
        "splitting_map": wh.plot_image(_splitting, title=_split_title,
                                       cmap=wh.coolwarm_white, center_zero=True),
        "redchi_map": wh.plot_image(fit_maps["redchi"], title="Reduced χ²", cmap="hot"),
    }
    wh.export_session_plots(data_path, _prefix, _figs)
    for _fig in _figs.values():
        wh.plt.close(_fig)
    _fit_record = {
        "init_params": _best_params,
        "n_peaks": n_peaks_ui.value,
        "roi_config": active_roi,
        "fit_range": list(fit_range_ui.value),
        "linear_bg": bool(linear_bg_ui.value),
        "x_bin": int(roi_xbin_input.value), "y_bin": int(roi_ybin_input.value),
        "bin_mode": roi_binmode_input.value,
        "per_pixel_init": bool(per_pixel_init_ui.value),
        "init_smooth_n": int(init_smooth_ui.value),
    }
    _disk_meta = wh.load_metadata(data_path)
    if _roi_name != "manual":
        _disk_meta.setdefault("rois", {}).setdefault(_roi_name, {}).setdefault("fits", {})[_fit_name] = _fit_record
    else:
        _disk_meta.setdefault("manual_fits", {})[_fit_name] = _fit_record
    wh.save_metadata(data_path, _disk_meta)
    set_refresh(lambda x: x + 1)
    mo.md(f"Exported **{_prefix}** — npz + 3 pdfs → `{wh.output_dir(data_path)}`")
    return


if __name__ == "__main__":
    app.run()
