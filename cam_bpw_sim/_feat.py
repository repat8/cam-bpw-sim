"""Feature detection."""

import abc
import collections as _col
import dataclasses as _dc
import warnings

import bpwave as _bp
import matplotlib.pyplot as _plt
import numpy as _np
import numpy.typing as _npt
import scipy.signal as _sg

from . import _sigutils as _su
from . import _version


@_dc.dataclass
class ChPointResult:
    """Base class for :class:`PointDetector` subclasses.

    May be extended to contain further algorithm-specific results.
    """

    chpoints: _bp.ChPoints
    """Detected points."""

    @property
    def onsets(self) -> _npt.NDArray[_np.int64]:
        """Shorthand property to get onsets, if available."""
        onsets = _np.array([cp.onset for cp in self.chpoints.indices if cp.onset >= 0])
        if not len(onsets):
            warnings.warn(f"{self.chpoints.alg} did not detect onsets")
        return onsets

    def __iter__(self):
        return iter(_dc.astuple(self))

    def plot(
        self,
        signal: _bp.Signal,
        ax: _plt.Axes | None = None,
        *,
        title: str = "",
        **kwargs,
    ) -> _plt.Axes:
        """Plots detected point on the signal, without having to insert them
        to the signal object.
        """
        if not ax:
            _, ax = _plt.subplots()
        assert ax  # mypy :(
        signal.plot(ax, label=signal.label or "signal", **kwargs)
        all_indices = _col.defaultdict(list)
        for ci in self.chpoints.indices:
            for name, index in ci.without_unset().items():
                all_indices[name].append(index)
        for name, indices in all_indices.items():
            ax.plot(signal.t[indices], signal.y[indices], "+", label=name)
        ax.legend()
        if title:
            ax.set(title=title)
        return ax


class ChPointAlg(abc.ABC, _su.AlgMeta):
    """Abstract base class of characteristic point detection algorithms.

    Implementations are callable objects, that can take additional parameters
    in the constructor::

        detector = FakePointDetector()
        result1 = detector(signal1)
        result2 = detector(signal2)

        detector = FakePointDetectorWithParams(threshold=0.9, a=42)
        result3 = detector(signal1)

    .. note::

        Subclasses should not store state other than the parameters passed,
        to make the callable object reusable for multiple signals.

        Subclasses should not modify the input signal.
    """

    @abc.abstractmethod
    def __call__(
        self,
        signal: _bp.Signal,
        *,
        show: bool = False,
    ) -> ChPointResult:
        """Run the algorithm.

        :param signal: filtered BP or PPG signal
        :param show: plot intermediary results (e. g. for debugging)
        :return: the result object containing the peaks and
            optionally other results.
        """


@_dc.dataclass
class LocMinDetector(ChPointAlg):
    """Onset detection by local minima."""

    order: int | None = None
    """Specifies a window, see :func:`scipy.signal.argrelextrema`."""

    refrac: float = 0.1
    """Refractory period in seconds, to avoid duplicate onsets."""

    def __call__(self, signal: _bp.Signal, *, show: bool = False) -> ChPointResult:
        order = int(signal.fs // 5) if self.order is None else self.order
        arg_rel_min = _np.squeeze(
            _sg.argrelextrema(signal.y, _np.less_equal, order=order)
        )
        i_refrac = int(_np.round(self.refrac * signal.fs))
        onsets = arg_rel_min[
            _np.nonzero(_np.diff(arg_rel_min, prepend=-i_refrac - 1) > i_refrac)[0]
        ]
        return ChPointResult(
            chpoints=_bp.ChPoints(
                indices=[_bp.CpIndices(onset=o) for o in onsets],
                alg=self.name,
                version=_version.__version__,
                params=self.params,
            )
        )


@_dc.dataclass
class ScipyFindPeaksResult(ChPointResult):
    all_peaks: _npt.NDArray[_np.int64]
    prominences: _npt.NDArray[_np.float64]


@_dc.dataclass
class ScipyFindPeaks(ChPointAlg):
    dt_onset: float = 0.5
    """[s] Minimum time distance between 2 consecutive onsets"""
    dt_sys_refl_dicr: float = 0.1
    """[s] Minimum time distance between the systolic, reflected
    and dicrotic peak"""
    dicr_search_end: float = 0.66
    """(0; 1] Signal length in which dicrotic peak can be found,
    relative to the full period length"""
    use_orig_onsets: bool = False
    """Don't detect onsets just detect the other points using
    the onsets stored in ``signal``"""
    ends_at_onset: bool = False
    """``True`` if we know that we cut the signal at an onset,
    so the last period is a full one.
    Otherwise it will be rejected, as not being followed by a rising edge.
    """

    def __call__(
        self,
        signal: _bp.Signal,
        *,
        show: bool = False,
    ) -> ScipyFindPeaksResult:
        assert signal.y.size > 0
        dy = _np.diff(signal.y)

        if self.use_orig_onsets:
            if signal.onsets.size == 0:
                raise ValueError("`use_orig_onsets` but `signal.onsets` is empty")
            onsets = signal.onsets
        else:
            # For rising edges, peak prominence is assumed to be the
            # half of the derivative value range
            rising, _ = _sg.find_peaks(
                dy,
                prominence=dy.max() / 2,
                distance=max(2, int(signal.fs * self.dt_onset)),
            )
            onsets_between_rising = [
                r1 + _np.argmin(signal.y[r1:r2]) for r1, r2 in zip(rising, rising[1:])
            ]
            first_onset = _np.argmin(signal.y[: rising[0]])
            onsets = _np.r_[first_onset, onsets_between_rising]
            if self.ends_at_onset:
                onsets = _np.r_[onsets, len(signal.y) - 1]

        indices: list[_bp.CpIndices] = []
        others = []
        all_prominences = []
        di_sys_refl_dicr = max(2, int(signal.fs * self.dt_sys_refl_dicr))
        for i, (onset, next_onset) in enumerate(zip(onsets, onsets[1:])):
            period = signal.y[onset:next_onset]
            sys = _np.argmax(period)
            absolute_sys = onset + sys
            falling = period[sys : int((next_onset - onset) * self.dicr_search_end)]
            peaks, props = _sg.find_peaks(
                falling, prominence=0, distance=di_sys_refl_dicr
            )
            # `prominence=0` is needed to make the result contain prominences
            d_peaks, d_props = _sg.find_peaks(
                _np.diff(falling), prominence=0, distance=di_sys_refl_dicr
            )

            refl_peak = refl_onset = dicr_peak = dicr_notch = -1
            if len(peaks):
                # If there are 2 peaks, they are the reflected and the
                # dicrotic peak respectively;
                # if there is 1, then it is the dicrotic peak.
                rel_dicr = peaks[0] if len(peaks) == 1 else peaks[1]
                rel_dicr_notch = _np.argmin(falling[:rel_dicr])
                dicr_peak = absolute_sys + rel_dicr
                dicr_notch = absolute_sys + rel_dicr_notch

                # Reflected peak is searched between the systolic peak and the
                # dicrotic notch, keeping some distance from both.
                d_peaks_cond = (di_sys_refl_dicr / 2 < d_peaks) & (
                    d_peaks < rel_dicr_notch - di_sys_refl_dicr / 2
                )
                d_refl = d_peaks[d_peaks_cond]
                d_refl_prominences = d_props["prominences"][d_peaks_cond]
                others += (d_refl + absolute_sys).tolist()
                all_prominences += d_refl_prominences.tolist()
                if d_refl.size:
                    rise_before_dicr = d_refl[_np.argmax(d_refl_prominences)]
                    maybe_refl_section = falling[rise_before_dicr:rel_dicr_notch]
                    maybe_refl = _np.argmax(maybe_refl_section)
                    refl_peak = absolute_sys + rise_before_dicr + maybe_refl
                    maybe_refl_onset = _np.argmin(signal.y[absolute_sys:refl_peak])
                    if maybe_refl_onset != refl_peak:
                        refl_onset = absolute_sys + maybe_refl_onset  # type: ignore

            indices.append(
                _bp.CpIndices(
                    onset=onset,
                    sys_peak=absolute_sys,
                    refl_onset=refl_onset,
                    refl_peak=refl_peak,
                    dicr_notch=dicr_notch,
                    dicr_peak=dicr_peak,
                )
            )

        result = ScipyFindPeaksResult(
            chpoints=_bp.ChPoints(
                alg=self.name,
                version=_version.__version__,
                params=self.params,
                indices=indices,
            ),
            all_peaks=_np.array(others, int),
            prominences=_np.array(all_prominences),
        )

        if show:
            _, axes = _plt.subplots(nrows=2, figsize=(25, 10))
            result.plot(
                signal,
                ax=axes[0],
                points=False,
                onsets=self.use_orig_onsets,
                legend="off",
            )
            axes[1].plot(signal.t[:-1], dy, lw=0.5, label="y'")
            axes[1].plot(
                signal.t[:-2],
                _np.diff(signal.y, 2) + dy.min(),
                lw=0.5,
                label="y''",
            )
            axes[1].set(xlim=(signal.t[0], signal.t[-1]))
            axes[1].legend()
            for p, prom in zip(result.all_peaks, result.prominences):
                t_peaks = signal.t[0] + p / signal.fs
                axes[0].axvline(
                    t_peaks,
                    c="k",
                    alpha=prom / result.prominences.max(),
                    lw=0.5,
                )
                axes[1].axvline(
                    t_peaks,
                    c="k",
                    alpha=prom / result.prominences.max(),
                    lw=0.5,
                )

        return result
