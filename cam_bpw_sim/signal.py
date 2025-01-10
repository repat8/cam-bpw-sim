"""Waveform processing.

* I/O
* Feature detection
* Baseline correction
* Normlization, denoising, resampling.
"""

from ._baseline import BaselineCorr, BaselineResult, CubicSplineCorr, SplineCorrResult
from ._feat import (
    ChPointAlg,
    ChPointResult,
    LocMinDetector,
    ScipyFindPeaks,
    ScipyFindPeaksResult,
)
from ._sigutils import (
    AlgMeta,
    SignalReader,
    TimestampedCsvReader,
    check_has_onsets,
    denoise,
    download_physionet,
    invert,
    load_signal,
    norm01,
    resample,
    save_signal,
    standardize,
    validate_timestamps,
)

__all__ = [
    e.__name__  # type: ignore[attr-defined]
    for e in [
        BaselineCorr,
        BaselineResult,
        CubicSplineCorr,
        SplineCorrResult,
        ChPointAlg,
        ChPointResult,
        LocMinDetector,
        ScipyFindPeaks,
        ScipyFindPeaksResult,
        AlgMeta,
        SignalReader,
        TimestampedCsvReader,
        check_has_onsets,
        denoise,
        download_physionet,
        invert,
        load_signal,
        norm01,
        resample,
        save_signal,
        standardize,
        validate_timestamps,
    ]
]
