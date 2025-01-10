"""Preprocessing steps for the validation module."""

import dataclasses as _dc
import itertools as _it

import bpwave as _bp
import bpwave.visu as _bpv
import matplotlib.pyplot as _plt
import numpy as _np
import numpy.typing as _npt
import pandas as _pd

from . import signal as _s
from . import utils as _u


def match_nominal_signal(
    nominal: _bp.Signal,
    /,
    measured: _bp.Signal,
    *,
    n_ccycles: int,
) -> _bp.Signal:
    """Transforms the nominal signal so that it match the sampling frequency
    of the measured signal.

    :param nominal: the nominal signal.
    :param measured: the measured signal.
    :param n_ccycles: number of cardiac cycles per full cam rotation.
    :return: the resampled nominal signal.
    """
    t_wheel_turn, _ = t_cam_rotation(measured, n_ccycles=n_ccycles)
    orig_t_scaled = _bp.Signal(
        y=nominal.y,
        unit="rel",
        t=_np.linspace(0, t_wheel_turn, len(nominal.y)),
        chpoints=nominal.chpoints,
    )
    orig_t_scaled = _s.resample(
        orig_t_scaled,
        t=_np.arange(int(_np.ceil(t_wheel_turn * measured.fs))) / measured.fs,
    )
    return orig_t_scaled


def split_combined_measurement(
    full: _bp.Signal,
    /,
    *,
    y_thresh: float = 0.6,
    t_region: float = 1.0,
    annot_full: bool = True,
) -> tuple[_bp.Signal, _bp.Signal]:
    """Splits a measurement consisting of noise (unactuated sensor) and signal
    measurement parts.

    Between the two, a section may be discarded.

    :param full: the full measurement time series.
    :param y_thresh: [0; 1], threshold for detection of the first rising edge,
        in terms of amplitude.
    :param t_region: [s], length of the region before the first rising edge,
        for finding the first onset.
    :param annot_full: add marks to ``full``.
    :return:
        0. noise-only section
        #. signal-only section with tolerace region ``t_region``
    """
    first_rising_edge = _np.nonzero(
        full.y > (full.y.min() + (full.y.max() - full.y.min()) * y_thresh)
    )[0][0]
    i_thresh = int(full.fs * t_region)
    search_start = _np.max([0, first_rising_edge - i_thresh])
    first_onset = search_start + full.y[search_start:first_rising_edge].argmin()
    noise_end = _np.max([0, first_onset - i_thresh])
    if annot_full:
        full.marks.update(
            dict(
                first_rising_edge=[first_rising_edge],
                first_onset=[first_onset],
                noise_end=[noise_end],
            )
        )
    return full[:noise_end], full[noise_end:]


def plot_combined_measurement(
    full: _bp.Signal,
    noise: _bp.Signal,
    signal: _bp.Signal,
    **subplots_kw,
) -> tuple[_plt.Axes, _plt.Axes, _plt.Axes]:
    """Visualizes splitting a combined measurement."""
    fig, (ax_f, ax_n, ax_s) = _plt.subplots(
        nrows=3, **(dict(figsize=(20, 8)) | subplots_kw)
    )
    full.plot(ax=ax_f, legend="off", title="Full measurement")
    noise.plot(ax=ax_n, legend="off", title="Noise section")
    signal.plot(ax=ax_s, legend="off", title="Signal section")
    fig.tight_layout()
    return ax_f, ax_n, ax_s


def plot_signal_slices(
    signal: _bp.Signal, /, **subplots_kw
) -> tuple[_plt.Axes, _plt.Axes]:
    """Visualizes :attr:`bpwave.Signal.slices`."""
    fig, (ax_sig, ax_slc) = _plt.subplots(
        nrows=2,
        sharex=True,
        **(dict(figsize=(15, 6), gridspec_kw={"height_ratios": [2, 1]}) | subplots_kw),
    )
    signal.plot(ax=ax_sig, legend="off")
    xlabel = ax_sig.get_xlabel()
    ax_sig.set(xlabel=None)
    for i, (name, slices) in enumerate(signal.slices.items()):
        for slc in slices:
            ax_slc.fill_between(
                [signal.t[slc.start], signal.t[slc.stop - 1]],
                [i - 0.4, i - 0.4],
                [i + 0.4, i + 0.4],
                alpha=0.5,
            )
    ax_slc.set(
        yticks=range(len(signal.slices)),
        yticklabels=signal.slices.keys(),
        xlabel=xlabel,
    )
    fig.tight_layout()
    return ax_sig, ax_slc


def t_cam_rotation(measured: _bp.Signal, /, n_ccycles: int) -> tuple[float, float]:
    """Measured time of 1 full cam rotation.

    :param measured: measured signal with onsets already detected.
    :param n_ccycles: number of cardiac cycles per full cam rotation.
    :return: (mean, std)
    :raises ValueError: if onsets are not present
    """
    _s.check_has_onsets(measured)
    t_rotations = _np.diff(measured.t_onsets[::n_ccycles])
    return t_rotations.mean(), t_rotations.std()


def correct_longterm_baseline_wander(
    signal: _bp.Signal,
    /,
    n_ccycles: int,
) -> tuple[_bp.Signal, _npt.NDArray[_np.float64], _npt.NDArray[_np.int64]]:
    """Removes longterm baseline wander (probably due to longterm change of
    sensor characteristics such as temperature or fatigue) using cubic spline
    interpolation based on the first onset of each full cam rotations.

    :param signal: measured signal with onsets already detected.
    :param n_ccycles: number of cardiac cycles per full cam rotation.
    :return:
        0. corrected signal
        #. interpolated baseline
        #. onsets selected for fitting the spline
    :raises ValueError: if onsets are not present
    """
    _s.check_has_onsets(signal)
    baseline_res = _s.CubicSplineCorr(step=n_ccycles)(signal)
    corrected = signal.copy(
        y=baseline_res.y_corr,
        label=signal.label and f"{signal.label} wo/ longterm wander",
    )
    corrected.meta["correct_longterm_baseline_wander"] = True
    return corrected, baseline_res.baseline, baseline_res.knots


def correct_baseline_wander(
    signal: _bp.Signal, /
) -> tuple[_bp.Signal, _npt.NDArray[_np.float64], _npt.NDArray[_np.int64]]:
    """Removes baseline wander using cubic spline interpolation.

    :param signal: measured signal with onsets already detected.
    :return:
        0. corrected signal
        #. interpolated baseline
        #. onsets selected for fitting the spline
    :raises ValueError: if onsets are not present
    """
    _s.check_has_onsets(signal)
    baseline_res = _s.CubicSplineCorr()(signal)
    corrected = signal.copy(
        y=baseline_res.y_corr,
        label=signal.label and f"{signal.label} wo/ baseline wander",
    )
    corrected.meta["correct_baseline_wander"] = True
    return corrected, baseline_res.y_corr, baseline_res.knots


def split_cam_rotations_corr(
    full: _bp.Signal,
    ref: _bp.Signal,
    *,
    tol: int,
    sync_window: int = 0,
    slice_key: str = "fcr",
    ext_slice_key: str = "fcr_ext",
    all_slice_key: str = "fcrs_all_ext",
    ccycle_slice_key: str = "cc_ref",
    ccycle_ext_slice_key: str = "cc_ref_ext",
) -> dict[str, list[slice]]:
    """Splits the full measured signal ``full`` to cam rotations by consecutively
    finding the reference (nominal or other ground truth) signal ``ref`` in it
    using cross-correlation.

    Cardiac cycles will be marked as well, using the onsets of the reference
    signal. The onsets of ``full`` are not considered in this step.

    .. note::
        ``full`` and ``ref`` are assumed to have close to uniform sampling,
        same sampling frequency and similar amplitude.
        These conditions are not validated.

    .. warning::
        Set ``sync_window`` long enough if ``full`` may start with a fractional
        rotation, otherwise the result may be incorrect.

    :param full: full measured signal (longer, than ``ref``)
    :param ref: reference signal
    :param tol: when calculating the correlation, a window of length
        ``tol + len(ref.y) + tol`` is used in each iteration.
        ``tol`` determines the tolerance of shifting the reference signal when
        looking for a matching region.
    :param sync_window: window increment for the first iteration.
        This may be longer than ``tol`` if ``full`` starts with a fractional
        rotation (cam was not started at the beginning of the nominal signal).
        If 0, we assume that ``full`` and ``ref`` are already synchronized at
        cardiac cycle precision.
        In the first iteration, the search window is
        ``tol + len(ref.y) + sync_window + tol`` long.
    :param slice_key: key of the full cam rotation slices
        (length same as ``ref``)
    :param ext_slice_key: key of the extended full cam rotation slices
        (each part has length of ``ref`` + 2*``tol``)
    :param all_slice_key: key of the slice marking the full processed range.
    :param ccycle_slice_key: key of the cardiac cycle slices.
    :param ccycle_ext_slice_key: key of the extended cardiac cycle slices.
    :returns: a ``dict`` with keys specified by ``slice_key``,
        ``ext_slice_key`` and ``all_slice_key``, containing the slices
        suitable for updating ``full.slices``.
    :raises ValueError: if *fs* and length conditions are violated
    """
    if not _np.isclose(full.fs, ref.fs):
        raise ValueError("`full` and `ref` must have the same sampling frequency")
    if len(full.y) < len(ref.y):
        raise ValueError("`full` must be longer than `ref`")
    if tol <= 0:
        raise ValueError("`tol` must be >= 0")

    len_ref = len(ref.y)
    len_full = len(full.y)
    len_rotation = len_ref + 2 * tol
    i_rot = 0
    idx = -tol

    ref_slices = list(ref.iter_ccycle_slices())

    fcr_slices_with_margin = []
    fcr_slices = []
    current_sync = sync_window
    while idx < len(full.y) - len_ref:
        rel_start, rel_stop = _align_with_corr(
            full.y[max(0, idx) : idx + tol + len_ref + current_sync + tol], ref.y
        )
        current_sync = 0  # Used in the first iteration only
        start = max(0, max(0, idx) + rel_start - tol)
        stop = min(len_full, max(0, max(0, idx) + rel_stop + tol))
        if stop - start == len_rotation:  # Collect only full rotations
            slc = _np.s_[start:stop]
            fcr_slices_with_margin.append(slc)
            fcr_slices.append(_np.s_[start + tol : stop - tol] if tol else slc)
        idx += rel_stop
        i_rot += 1

    cc_slices = [
        slice(cc_slc.start + fcr_slc.start, cc_slc.stop + fcr_slc.start)
        for fcr_slc in fcr_slices
        for cc_slc in ref_slices
    ]
    return {
        ext_slice_key: fcr_slices_with_margin,
        slice_key: fcr_slices,
        all_slice_key: [
            slice(fcr_slices_with_margin[0].start, fcr_slices_with_margin[-1].stop)
        ],
        ccycle_slice_key: cc_slices,
        ccycle_ext_slice_key: (
            [slice(slc.start - tol, slc.stop + tol) for slc in cc_slices]
            if tol
            else cc_slices
        ),
    }


def align_ccycles_corr(
    full: _bp.Signal,
    ref: _bp.Signal,
    *,
    tol: int,
    section: slice | None = None,
    slice_key: str = "cc_aligned",
    ext_slice_key: str = "cc_aligned_ext",
) -> dict[str, list[slice]]:
    """Aligns cardiac cycles to the reference signal using cross-correlation.

    It is assumed that the first cardiac cycle of ``full[section]`` corresponds to
    the first one of ``ref`` and there are at least ``tol`` data points before
    the first onset of ``full[section]``.

    :param full: full measured signal (longer, than ``ref``)
    :param ref: reference signal
    :param tol: when calculating the correlation, a window of length
        ``tol + len(ref.y) + tol`` is used in each iteration.
        ``tol`` determines the tolerance of shifting the reference signal when
        looking for a matching region.
    :param section: process only the section within this slice.
    :param slice_key: key of the aligned cardiac cycle slices
        (length same as ``ref``)
    :param ext_slice_key: key of the extended aligned cardiac cycle slices
        (each part has length of ``ref`` + 2*``tol``)
    :return: a ``dict`` with keys specified by ``slice_key``,
        ``ext_slice_key`` containing the slices suitable for updating
        ``full.slices``.
    """
    _s.check_has_onsets(full, "full")
    _s.check_has_onsets(ref, "ref")
    if not _np.isclose(full.fs, ref.fs):
        raise ValueError("`full` and `ref` must have the same sampling frequency")

    if section:
        full = full[section]

    if len(full.y) < len(ref.y):
        raise ValueError("`full` must be longer than `ref`")
    if tol <= 0:
        raise ValueError("`tol` must be >= 0")

    onset = full.onsets[0]
    if onset < tol:
        raise ValueError(f"First onset of `full` ({onset}) must be >= `tol` ({tol}).")

    ref_slices = list(ref.iter_ccycle_slices())

    cc_slices_with_margin = []
    cc_slices = []
    offset = section.start if section else 0

    for ref_cc_slc, cc_slc in zip(_it.cycle(ref_slices), full.iter_ccycle_slices()):
        cc_ref = ref.y[ref_cc_slc]
        cc_full = full.y[cc_slc.start - tol : cc_slc.stop + tol]
        rel_start, rel_stop = _align_with_corr(cc_full, cc_ref)
        start = offset + cc_slc.start - tol + rel_start
        stop = offset + cc_slc.start - tol + rel_stop
        cc_slices_with_margin.append(_np.s_[start - tol : stop + tol])
        cc_slices.append(_np.s_[start:stop])

    return {
        slice_key: cc_slices,
        ext_slice_key: cc_slices_with_margin,
    }


def _align_with_corr(full: _np.ndarray, ref: _np.ndarray) -> tuple[int, int]:
    cvo = _np.correlate(full, ref, mode="same")
    cvo_max = _np.argmax(cvo).item()
    shift = cvo_max - len(ref) // 2
    return shift, shift + len(ref)


@_dc.dataclass(kw_only=True)
class PreprocResult:
    """Result of :func:`preproc_for_validation`."""

    nominal_matched: _bp.Signal
    """Nominal signal matched to the measured wrt. :math:`f_s`."""

    measured_long_bw_corr: _bp.Signal

    measured_bw_corr: _bp.Signal


def preproc_for_validation(
    measured: _bp.Signal,
    raw_nominal: _bp.Signal,
    *,
    n_ccycles: int,
    t_tol: float = 0.1,
    t_sync_window: float | None = None,
) -> PreprocResult:
    """High level function to preprocess measured signals for evaluation
    statistics.

    :param measured: full measured signal
    :param raw_nominal: raw nominal signal (original amplitude and :math:`f_s`)
    :param n_ccycles: number of cardiac cycles per full cam rotation.
    :param t_tol: [s], tolerance of correlation window,
        see :func:`split_cam_rotations_corr`.
    :param t_sync_window: [s], length of sychronization window, default is half
        length of the nominal signal. See :func:`split_cam_rotations_corr`.
    :return: preprocessed signals
    """
    _s.check_has_onsets(measured, "measured")
    _s.check_has_onsets(raw_nominal, "raw_nominal")

    nominal_matched = match_nominal_signal(raw_nominal, measured, n_ccycles=n_ccycles)
    nom_y_n01 = _s.norm01(nominal_matched.y)
    tol = int(t_tol * nominal_matched.fs)

    (
        measured_lbw,
        lbw,
        lbw_points,
    ) = correct_longterm_baseline_wander(measured, n_ccycles=n_ccycles)
    # The ends are discarded as boundary effects are likely
    measured_lbw = measured_lbw[
        measured_lbw.onsets[n_ccycles] - tol : measured_lbw.onsets[-n_ccycles] + tol
    ]

    nominal_matched.y = nom_y_n01 * (measured_lbw.y.max() - measured_lbw.y.min())
    nominal_matched.unit = "rel"
    sync_window = (
        len(nominal_matched.y) // 2
        if t_sync_window is None
        else int(t_sync_window * nominal_matched.fs)
    )

    measured_lbw.slices |= split_cam_rotations_corr(
        measured_lbw,
        nominal_matched,
        tol=tol,
        sync_window=sync_window,
    )
    measured_lbw.slices |= align_ccycles_corr(
        measured_lbw,
        nominal_matched,
        tol=tol // 2,
        section=measured_lbw.slices["fcrs_all_ext"][0],
    )

    measured_bw, bw, bw_points = correct_baseline_wander(measured)
    # The ends are discarded as boundary effects are likely
    measured_bw = measured_bw[
        measured_bw.onsets[n_ccycles] - tol : measured_bw.onsets[-n_ccycles] + tol
    ]
    nominal_matched.y = nom_y_n01 * (measured_bw.y.max() - measured_bw.y.min())
    measured_bw.slices |= split_cam_rotations_corr(
        measured_bw,
        nominal_matched,
        tol=tol,
        sync_window=sync_window,
    )
    measured_bw.slices |= align_ccycles_corr(
        measured_bw,
        nominal_matched,
        tol=tol // 2,
        section=measured_bw.slices["fcrs_all_ext"][0],
    )

    return PreprocResult(
        nominal_matched=nominal_matched,
        measured_long_bw_corr=measured_lbw,
        measured_bw_corr=measured_bw,
    )


def build_fcr_array(
    signal: _bp.Signal,
    fcr_key: str,
) -> _u.NDArray2D[_bp.Signal]:  # type: ignore[type-var]
    """Utility to cut a signal to full cam rotations with zeroed timestamps.

    See :func:`split_cam_rotations_corr` on how to generate FCR slices.

    .. note::
        This function creates a column vector to simplify cross comparisons.

    :param signal: signal with marked FCRs.
    :param fcr_key: key of the stored slices.
    :return: N×1 array of FCRs
    """
    return _np.array([[signal[slc].shift_t()] for slc in signal.slices[fcr_key]])


def build_ccycle_matrix(
    signal: _bp.Signal,
    ccycle_key: str,
    n_ccycles: int,
) -> _u.NDArray2D[_bp.Signal]:
    """Utility to cut a signal to cardiac cycles with zeroed timestamps.

    See :func:`split_cam_rotations_corr` on how to generate FCR and cardiac
    cycle slices.

    :param signal: signal with marked FCRs.
    :param ccycle_key: key of the stored slices.
    :param n_ccycles: number of cardiac cycles per FCR.
    :return: N×n_ccycles array of cardiac cycles.
    """
    return _np.array(
        [
            [signal[slc].shift_t() for slc in signal.slices[ccycle_key][i::n_ccycles]]
            for i in range(n_ccycles)
        ]
    ).T


def build_chpoints_table(
    signal: _bp.Signal,
    fcr_key: str = "fcr_ext",
    *,
    include_end_onset: bool = False,
) -> _pd.DataFrame:
    """Constructs a :class:`pandas.DataFrame` of the characteristic points
    of the aligned extended full cam rotations.

    See :func:`preproc_for_validation` and :func:`split_cam_rotations_corr` on
    how to generate full cam rotation slices.

    :param signal: signal with:attr:`Signal.slices` filled.
    :param fcr_key: slice key of extended full cam rotations.
    :param include_end_onset: include the first onset of the next cardiac cycle.
    :return: a :class:`pandas.DataFrame` of characteristic points.
    """
    slices = signal.slices[fcr_key]

    records = []
    for k_fcr, slc in enumerate(slices):
        fcr = signal[slc].shift_t()
        assert fcr.chpoints is not None

        indices = (
            fcr.chpoints.indices if include_end_onset else fcr.chpoints.indices[:-1]
        )
        for k_ccycle, cp in enumerate(indices):
            cpi = cp.without_unset()
            record = {"fcr": k_fcr, "ccycle": k_ccycle}
            record |= cpi
            record |= {f"t_{name}": fcr.t[i] for name, i in cpi.items()}
            record |= {f"y_{name}": fcr.y[i] for name, i in cpi.items()}
            records.append(record)

    return _pd.DataFrame.from_records(records)


def match_sampling(
    signals: _u.Array1D[_bp.Signal],
    select: str = "lowest",
) -> tuple[list[_bp.Signal], int]:
    """Selects the signal with the lowest/highest sampling frequency and resamples
    the others using spline interpolation evaluated at the time points of it.
    (The resulting signals will have the same number of points).

    :param signals: signals representing the same measurement period
        (previously aligned)
    :param select: ``'lowest'`` or ``'highest'``.
        Reference selection criterion. ``'lowest'`` is the recommended for
        less interpolation error.
    :return: (resampled signals, index of the reference signal in ``signals``)
    """
    assert select in ("lowest", "highest")

    if select == "highest":
        fn = max
    elif select == "lowest":
        fn = min
    else:
        raise ValueError('Parameter `select` must be "lowest"/"highest"')

    i_ref: int = fn(enumerate(signals), key=lambda x: x[1].fs)[0]  # type: ignore
    reference = signals[i_ref]
    t_ref = reference.t

    results = []
    for i, signal in enumerate(signals):
        if i == i_ref:
            result = reference
        else:
            result = _s.resample(signal, t=t_ref)
        results.append(result)
    return results, i_ref


def plot_stacked_slices(
    signal: _bp.Signal,
    slices: str | _u.Array1D[slice],
    *,
    desc: str = "",
    overlay: _bp.Signal | None = None,
    ax: _plt.Axes | None = None,
    **plot_kws,
) -> _plt.Axes:
    """Plots signal slices overlayed on each other.

    :param signal: signal with :attr:`Signal.slices` filled.
    :param slices: slice key or slice list.
    :param desc: part of plot title.
    :param overlay: signal to plot on the top of others.
    :param ax: axes for the plot.
    :param plot_kws: args to pass to :meth:`bpwave.Signal.plot`.
    :raises KeyError: if ``key`` is invalid.
    """
    if isinstance(slices, str):
        slices_ = signal.slices[slices]
    else:
        slices_ = slices  # mypy!

    ax = _u.ensure_ax(ax)
    plot_kws = plot_kws or {}
    plot_kws["legend"] = "off"
    for slc in slices_:
        signal[slc].shift_t().plot(ax=ax, **plot_kws)

    if overlay:
        overlay.plot(ax, "r", **plot_kws)

    ax.set(title=f"{desc}{'; ' if desc else ''}N={len(slices_)}")
    ax.grid(True)

    return ax


def plot_stacked_ccycles(
    signal: _bp.Signal,
    slices: str,
    *,
    title: str = "",
    n_ccycles: int,
    overlay: _u.Array1D[_bp.Signal | None] | None = None,
) -> _u.Array1D[_plt.Axes]:
    """Plots cardiac cycles overlayed on each other.

    :param signal: signal with :attr:`Signal.slices` filled.
    :param slices: slice key of cardiac cycles.
    :param title: figure title.
    :param n_ccycles: number of cardiac cycles per full cam rotation.
    :param overlay: e. g. reference cardiac cycles, ``len(overlay) == n_ccycles``.
    :return: subplots
    """
    if overlay:
        assert len(overlay) == n_ccycles
    else:
        overlay = [None] * n_ccycles
    with _bpv.figure(autogrid=n_ccycles, title=title) as (
        _,
        axes,
    ):
        axes = axes.ravel()
        for i, ax in enumerate(axes):
            plot_stacked_slices(
                signal,
                signal.slices[slices][i::n_ccycles],
                desc=f"Cardiac cycle {i}",
                overlay=overlay[i],
                ax=ax,
                onsets=False,
            )
    return axes
