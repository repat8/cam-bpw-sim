"""Baseline correction."""

import abc
import dataclasses as _dc

import bpwave as _bp
import bpwave.visu as _bpv
import matplotlib.pyplot as _plt
import numpy as _np
import numpy.typing as _npt
import scipy.interpolate as _ipl

from . import _sigutils as _su


@_dc.dataclass
class BaselineResult:
    """Result class for baseline correction algorithms."""

    y_corr: _npt.NDArray[_np.float64]
    """The baseline-corrected ``y`` values of the signal."""

    alg: str
    """Name of the baseline correction algorithm."""

    def plot(
        self,
        signal: _bp.Signal,
        ax: _plt.Axes | None = None,
        *,
        title: str | None = None,
    ) -> _plt.Axes:
        """Plot baseline-corrected signal on the input signal.

        :return: the axes object
        """
        if not ax:
            _, ax = _plt.subplots()
        assert ax  # mypy :(
        signal.plot(ax, label=signal.label or "raw")
        _bpv.plot_signal(ax, signal.t, self.y_corr, append=True, label=self.alg)
        ax.legend()
        if title:
            ax.set(title=title)
        return ax


@_dc.dataclass
class BaselineCorr(abc.ABC, _su.AlgMeta):
    """Abstract base class of baseline correction algorithms.

    Implementations are callable objects, that can take additional parameters
    in the constructor::

        corr = FakeBaselineCorr()
        result1 = corr(signal1)
        result2 = corr(signal2)

        corr = FakeBaselineCorrWithParams(threshold=0.9, a=42)
        result3 = corr(signal1)

    .. note::

        Subclasses should not store state other than the parameters passed,
        to make the callable object reusable for multiple signals.

        Subclasses should not modify the input signal.

        :attr:`BaselineResult.y_corr` must have the same length as the input.
    """

    @abc.abstractmethod
    def __call__(self, signal: _bp.Signal, *, show: bool = False) -> BaselineResult:
        """Run the algorithm.

        :param signal: filtered BP or PPG signal
        :param show: plot intermediary results (e. g. for debugging)
        :return: the result object containing the corrected signal and
            optionally other results.
        """


@_dc.dataclass
class SplineCorrResult(BaselineResult):
    """Results of :class:`SplineCorr`."""

    baseline: _npt.NDArray[_np.float64]
    """Baseline interpolated with cubic spline."""

    spline: _ipl.CubicSpline
    """The cubic spline fit on the onsets."""

    knots: _npt.NDArray[_np.int64]
    """Knots of the spline (a subset of onsets)."""


@_dc.dataclass(kw_only=True)
class CubicSplineCorr(BaselineCorr):
    """Baseline correction by subtracting the baseline estimated by a cubic
    spline fit on the onset points.
    """

    step: int = 1
    """Fit spline to every ``step`` th onset."""

    def __call__(self, signal: _bp.Signal, *, show: bool = False) -> SplineCorrResult:
        _su.check_has_onsets(signal)
        i_onsets = signal.onsets[:: self.step]
        y_onsets = signal.y[i_onsets]
        spline = _ipl.CubicSpline(x=signal.t[i_onsets], y=y_onsets)
        base = spline(signal.t)
        corrected = signal.y - base
        if show:
            with _bpv.figure(block=True) as (_, axes):
                axes[0].plot(signal.y)
                axes[0].plot(i_onsets, y_onsets, "r+")
                axes[0].plot(base)
        return SplineCorrResult(
            y_corr=corrected,
            alg=self.name,
            baseline=base,
            spline=spline,
            knots=i_onsets,
        )
