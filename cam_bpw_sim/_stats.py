"""Statistics used for validation."""

import dataclasses as _dc
import typing as _t

import bpwave as _bp
import bpwave.visu as _bpv
import h5py
import matplotlib.pyplot as _plt
import numpy as _np
import numpy.typing as _npt
import pandas as _pd
import scipy.stats
import seaborn as _sns

from . import _preproc
from . import utils as _u

ArrayComparator: _t.TypeAlias = _t.Callable[[_u.Array1D, _u.Array1D], float]
SignalComparator: _t.TypeAlias = _t.Callable[[_bp.Signal, _bp.Signal], float]


@_dc.dataclass
class CrossCompareResult:
    """Result of :func:`cross_compare`."""

    STATS_COLS: _t.ClassVar[list[str]] = [
        "count",
        "mean",
        "std",
        "min",
        "p25",
        "p50",
        "p75",
        "max",
    ]

    data: _pd.DataFrame
    """Result of the comparison in long-form :class:`pandas.DataFrame`."""

    measured: _u.NDArray2D[_bp.Signal]  # type: ignore[type-var]
    """Reference to the compared signal sections."""

    meas_or_ref: _u.NDArray2D[_bp.Signal]  # type: ignore[type-var]
    """Reference to the reference signal sections."""

    def stats(self) -> _pd.DataFrame:
        """Descriptive statistics of the comparison."""
        df = self.data.loc[:, 0:].describe().T
        df.columns = self.STATS_COLS
        return df

    def stats_to_hdf(self, group: h5py.Group, name: str) -> h5py.Dataset:
        """Saves statistics to HDF5 file along with the attributes:

        ``columns``
            columns of the statistics dataset
        ``n_fcrs``
            number of full cam rotations
        ``n_ccycles``
            number of cardiac cycles per rotation

        :param group: HDF5 file or group.
        :param name: the name of the new dataset.
        :return: the new dataset.
        """
        ds = group.create_dataset(name, data=self.stats().to_numpy())
        ds.attrs["columns"] = self.STATS_COLS
        ds.attrs["n_fcrs"] = self.measured.shape[0]
        ds.attrs["n_ccycles"] = self.measured.shape[1]
        return ds

    def boxplot(self, value_label: str | None = None, **kwargs) -> _plt.Axes:
        """Draws a boxplot from the statistics.

        :param value_label: label of the non-categorical axis
        :param kwargs: arguments to be passed to :meth:`pandas.DataFrame.plot.box`.
        :return: the :class:`Axes` of the plot.
        """
        multicol = self.meas_or_ref.shape[1] > 1
        values_label = (
            f"Amplitude [{self.measured[0, 0].unit}]"
            if value_label is None
            else value_label
        )
        ccycles_label = "Cardiac cycle"
        dft_kwargs = dict(
            showmeans=True,
            vert=multicol,
            ylabel=values_label if multicol else ccycles_label,
            xlabel=ccycles_label if multicol else values_label,
            grid=True,
        )
        dft_kwargs |= kwargs
        return self.data.loc[:, 0:].plot.box(**dft_kwargs)

    def matrix(self, ccycle: int) -> _pd.DataFrame:
        """Comparisons of a cardiac cycle ``ccycle`` in form of a matrix
        (pivot table).

        :param ccycle: index of cardiac cycle.
        :return: pivot table.
        """
        return self.data.pivot(index="fcr_m", columns="fcr_r", values=ccycle)

    def heatmaps(self, **kwargs) -> _u.Array1D[_plt.Axes]:
        """Plots :attr:`data` in form of heatmaps.

        :param kwargs: notably ``figsize`` and ``title``.
        :return: the :class:`Axes` of the plot.
        """
        n_ccycles = self.measured.shape[1]
        all_results = self.data.loc[:, 0:]
        vmin = all_results.min(axis=None)
        vmax = all_results.max(axis=None)
        with _bp.visu.figure(autogrid=n_ccycles, **kwargs) as (_, axes):
            axes = axes.ravel()
            for i_cc, ax in enumerate(axes):
                _sns.heatmap(
                    self.matrix(i_cc), ax=ax, square=True, vmin=vmin, vmax=vmax
                )
                if n_ccycles > 1:
                    ax.set(title=f"Cardiac cycle {i_cc}")
                ax.grid(False)
        return axes

    def examples(
        self,
        meas_desc: str,
        comp_desc: str,
        *,
        good: _t.Literal["min", "max"],
        figsize: tuple[int, int] = (10, 10),
    ) -> None:
        """Visualizes the result of cross comparison by plotting the waveform of
        the best, mean and worst match.

        :param meas_desc: description of the measurement for the figure title.
        :param comp_desc: description of the comparison for the figure title.
        :param good: ``"min"`` or ``"max"``, depending on the metrics used to fill
            the matrices.
        :param figsize: full figure size.
        """
        assert good in ("min", "max")
        n_ccycles = self.measured.shape[1]

        def plot_ex(
            i_fcr_m: int, i_fcr_r: int, i_cc: int, ax: _plt.Axes, label: str
        ) -> None:
            plot_kws = dict(
                ax=ax, legend="off", onsets=False, points=False, marks=False
            )
            (a := self.measured[i_fcr_m, i_cc]).plot(**plot_kws)
            (b := self.meas_or_ref[i_fcr_r, i_cc]).plot(**plot_kws)
            value = self.data.loc[
                (self.data["fcr_m"] == i_fcr_m) & (self.data["fcr_r"] == i_fcr_r), i_cc
            ].item()
            ax.set(
                title=(
                    f"{label}: {value:.4f}; " f"Rot. {i_fcr_m} vs {i_fcr_r}, cc. {i_cc}"
                ),
                xlim=(0, max(a.t[-1], b.t[-1])),
            )

        with _bpv.figure(
            nrows=3,
            ncols=n_ccycles,
            title=f"{meas_desc}{meas_desc and comp_desc and '; '}{comp_desc}",
            figsize=figsize,
            sharey=True,
        ) as (_, axes):
            axes = _np.c_[axes]
            fcr_indices = self.data[["fcr_m", "fcr_r"]]
            for i_cc in range(n_ccycles):
                values = self.data[i_cc]
                i_min = fcr_indices.loc[values.idxmin()]
                i_max = fcr_indices.loc[values.idxmax()]
                i_mean_m, i_mean_r = fcr_indices.loc[
                    (values - values.mean()).abs().idxmin()
                ]

                if good == "max":
                    i_best_m, i_best_r = i_max
                    i_worst_m, i_worst_r = i_min
                else:
                    i_best_m, i_best_r = i_min
                    i_worst_m, i_worst_r = i_max

                plot_ex(i_best_m, i_best_r, i_cc, axes[0, i_cc], "Best")
                plot_ex(i_mean_m, i_mean_r, i_cc, axes[1, i_cc], "Mean")
                plot_ex(i_worst_m, i_worst_r, i_cc, axes[2, i_cc], "Worst")


def plot_observed_values(
    measured: _bp.Signal, /, **fig_kws
) -> tuple[_plt.Figure, _u.Array1D[_plt.Axes], _np.ndarray]:
    """Plots information an the distribution of measured values
    (empirical cumulative distribution functions (ECDF) and
    distance of unique values).

    :return:
        0. figure
        #. axes of the subplots
        #. unique measured values
    """
    unique_y = _np.unique(measured.y)
    fig, axes = _plt.subplots(ncols=2, **(dict(figsize=(15, 5)) | fig_kws))
    _sns.ecdfplot(unique_y, stat="proportion", ax=axes[0], lw=0.5)
    axes[0].set(title=f"ECDF (min={unique_y.min():.4f}; max={unique_y.max():.4f})")
    axes[1].plot(_np.diff(unique_y), lw=0.5)
    axes[1].set(title=f"Distance of unique values (N={len(unique_y)})")
    for ax in axes:
        ax.grid(True)
    fig.tight_layout()
    return fig, axes, unique_y


def cross_compare_mat(
    a: _u.Array1D[_bp.Signal],
    b: _u.Array1D[_bp.Signal],
    /,
    compare: SignalComparator,
    *,
    skip_diag: bool = True,
    symmetric: bool = True,
) -> _npt.NDArray[_np.float64]:
    """Evaluates ``compare`` on all combinations of the elements of ``a`` and ``b``.

    :param a: 1D array of signals.
    :param b: 1D array of signals.
    :param compare: function to evaluate on each element pair of ``a`` and ``b``.
    :param skip_diag: leave diagonal elements as :const:`numpy.nan`,
        ``compare`` won't be called there.
        Ignored if ``a`` and ``b`` have different length.
    :param symmetric: if ``compare`` is symmetric, computation is optimized.
    :return: ``len(a)`` × ``len(b)`` matrix

    .. seealso::
        :func:`plot_cross_compare_examples`
            Plots example waveforms.
        :func:`plot_cross_compare_mats`
            Plots the comparison result matrix.
    """
    square = len(a) == len(b)
    if not square and skip_diag:
        skip_diag = False
    matrix = _np.zeros((len(a), len(b)))
    for i, t1 in enumerate(a):
        for j, t2 in enumerate(b):
            if i == j and skip_diag:
                continue
            if j > i and symmetric:
                continue
            matrix[i, j] = compare(t1, t2)
    if square:
        if skip_diag:
            _np.fill_diagonal(matrix, _np.nan)
        if symmetric:
            _u.mirror_to_diag(matrix)
    return matrix


def cross_compare(
    measured: _u.Array2D[_bp.Signal],
    meas_or_ref: _u.Array2D[_bp.Signal],
    /,
    compare: SignalComparator,
    *,
    symmetric: bool = True,
    self_compare: bool = False,
) -> CrossCompareResult:
    """Evaluates ``compare`` on all combinations of the elements of ``measured``
    and ``meas_or_ref``.

    :param measured: (N×M) array of signal sections, where N is the number of
        full cam rotations and M is the number of cardiac cycles on the cam.
    :param meas_or_ref: (N×M) or (1×M) array of signal sections, to which the
        ones in ``measured`` will be compared.
        Use (1×M) array for comparison to the nominal signal.
    :param compare: a metric to evaluate on each element pair
    :param symmetric: if ``True``, ``compare`` will only be evaluated with one
        parameter order.
    :param self_compare: if ``True``, signal sections will be compared with
        themselves as well.
    :return: a result object with a long-form table of the comparison results
        (see :attr:`CrossCompareResult.data`).
    """
    measured = _np.asarray(measured, dtype=_np.object_)
    meas_or_ref = _np.asarray(meas_or_ref, dtype=_np.object_)
    if measured.ndim != 2 or meas_or_ref.ndim != 2:
        raise ValueError("`measured` and `meas_or_ref` must be 2D")
    n_fcrs_m, n_ccycles_m = measured.shape
    n_fcrs_r, n_ccycles_r = meas_or_ref.shape
    if n_ccycles_m != n_ccycles_r:
        raise ValueError(
            f"`measured` and `meas_or_ref` must contain the same # of "
            f"cardiac cycles (got {n_ccycles_m} and {n_ccycles_r})"
        )
    if n_fcrs_r not in (n_fcrs_m, 1):
        raise ValueError(
            "The # of full cam rotations in `meas_or_ref` must be "
            "either 1 or the same as in `measured`"
        )
    if n_fcrs_r == 1:
        # Comparison to a reference, so "diagonal" (i. e. first element) needed
        self_compare = True

    n_rows = n_fcrs_m * n_fcrs_r
    if symmetric:
        n_rows = _np.ceil(n_fcrs_m * (n_fcrs_r + 1) / 2)
    if not self_compare:
        n_rows -= n_fcrs_r
    results = _np.full((int(n_rows), 2 + n_ccycles_r), _np.nan)
    i = 0
    for k_fcr_m, fcr_m in enumerate(measured):
        for k_fcr_r, fcr_r in enumerate(meas_or_ref):
            if k_fcr_m < k_fcr_r and symmetric:
                continue
            if k_fcr_m == k_fcr_r and not self_compare:
                continue
            results[i, :2] = [k_fcr_m, k_fcr_r]
            for k_cc, (cc_meas, cc_ref) in enumerate(zip(fcr_m, fcr_r)):
                results[i, 2 + k_cc] = compare(cc_meas, cc_ref)
            i += 1

    df = _pd.DataFrame(
        results,
        columns=["fcr_m", "fcr_r"] + list(range(n_ccycles_r)),
    ).astype({"fcr_m": int, "fcr_r": int})

    return CrossCompareResult(df, measured, meas_or_ref)


def with_resampling(
    compare: _t.Callable[[_np.ndarray, _np.ndarray], float]
) -> _t.Callable[[_bp.Signal, _bp.Signal], float]:
    def compare2(a: _bp.Signal, b: _bp.Signal) -> float:
        [resamp1, resamp2], _ = _preproc.match_sampling([a, b])
        return compare(resamp1.y, resamp2.y)

    return compare2


def with_crop(
    compare: _t.Callable[[_np.ndarray, _np.ndarray], float]
) -> _t.Callable[[_bp.Signal, _bp.Signal], float]:
    def compare2(a: _bp.Signal, b: _bp.Signal) -> float:
        assert _np.isclose(a.fs, b.fs), f"{a.fs=}, {b.fs=}"
        n = min(len(a.y), len(b.y))
        return compare(a.y[:n], b.y[:n])

    return compare2


def a2s_comp(compare: ArrayComparator) -> SignalComparator:
    """Transforms an array comparator to a signal comparator."""

    def scompare(query: _bp.Signal, ref: _bp.Signal) -> float:
        return compare(query.y, ref.y)

    return scompare


def rmse(query: _u.Array1D, ref: _u.Array1D) -> float:
    r"""Root mean squared error of 2 equal length signals.

    .. math::

        \text{RMSE}(y, y_0) = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - y_{0_i})^2}
    """
    _check_equal_length(query, "query", ref, "ref")
    e = _np.asarray(query, dtype=float) - _np.asarray(ref, dtype=float)
    e *= e
    e /= _np.size(query)
    e = _np.sqrt(_np.nansum(e))
    return e


rmse_s = a2s_comp(rmse)
"""Root mean squared error of 2 equal length signals."""


def nrmse(query: _u.Array1D, ref: _u.Array1D) -> float:
    r"""Normalized RMSE.

    .. math::

        \text{NRMSE}(y, y_0) = \frac{\text{RMSE}(y, y_0)}{y_{max} - y_{min}}
    """
    return rmse(query, ref) / (_np.max(query) - _np.min(query))


nrmse_s = a2s_comp(nrmse)
"""Normalized RMSE."""


def pearson(query: _u.Array1D, ref: _u.Array1D) -> float:
    """Perason correlation coefficient of 2 equal length signals."""
    _check_equal_length(query, "query", ref, "ref")
    return scipy.stats.pearsonr(query, ref).correlation


pearson_s = a2s_comp(pearson)
"""Perason correlation coefficient of 2 equal length signals."""


def _check_equal_length(a, a_name, b, b_name) -> None:
    if (a_s := _np.size(a)) != (b_s := _np.size(b)):
        raise ValueError(
            f"`{a_name}` and `{b_name}` must have equal length, " f"got {a_s} and {b_s}"
        )
