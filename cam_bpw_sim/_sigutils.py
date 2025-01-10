"""Signal processing utils."""

import abc
import dataclasses as _dc
import datetime as _dt
import json
import pathlib as _pl
import typing as _t

import bpwave as _bp
import h5py
import numpy as _np
import pandas as _pd
import scipy.interpolate as _ipl
import scipy.signal as _sg
import wfdb

from . import utils as _u


class AlgMeta:
    """Algorithm metadata for documentation purposes in HDF5 files."""

    @property
    def name(self) -> str:
        """Qualified name of the algorithm."""
        return f"{self.__module__}.{type(self).__name__}"

    @property
    def params(self) -> dict[str, _t.Any]:
        """Parameter dict."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class SignalReader(abc.ABC, AlgMeta):
    """Base functor class for file to :class:`Signal` converters.

    Subclasses may take additional parameters needed for the specific file
    format, e. g.::

        c = FakeCsvConverter(y_column='Pressure', t_column='Timestamp')
        signal, _ = c('read/from/here.csv')

    Implementations should override :meth:`_read`.
    """

    validate_t: bool = True
    """Validate timestamps."""

    def __call__(
        self,
        in_path: _u.PathLike,
        *,
        rel_to: _u.PathLike | None = None,
    ) -> tuple[_bp.Signal, _t.Any]:
        """Reads file at ``in_path`` and constructs a :class:`Signal` from it.

        :param in_path: path of the input file
        :param rel_to: path part not to be included in metadata
        :return: the signal object and implementation-specific other data
        """
        conv_in_path = _pl.Path(in_path)
        signal, other = self._read(conv_in_path)
        if self.validate_t:
            validate_timestamps(signal.t, check_unique=True)
        signal.meta["source_file"] = str(
            conv_in_path if not rel_to else conv_in_path.relative_to(rel_to)
        )
        signal.meta["source_file_date"] = _dt.datetime.now().isoformat()
        signal.meta["source_file_reader"] = self.name
        signal.meta["source_file_params"] = json.dumps(self.params)
        return signal, other

    @abc.abstractmethod
    def _read(self, in_path: _pl.Path) -> tuple[_bp.Signal, _t.Any]:
        """Performs the conversion."""


@_dc.dataclass
class TimestampedCsvReader(SignalReader):
    """Reader for simple timestamp - value pair CSV files."""

    t_column: str | int
    """Name or index (if ``not has_header``) of column."""

    y_column: str | int
    """Name or index (if ``not has_header``) of column."""

    _: _dc.KW_ONLY

    delimiter: str = ","
    """Cell delimiter."""

    has_header: bool = True
    """Whether first line is header."""

    skiprows: int = 0
    """Skip top lines."""

    comment: str | None = "#"
    """Comment mark at line start."""

    t_format: str = "%H:%M:%S.%f"
    """:meth:`datetime.datetime.strptime` time format or ``'float'``."""

    unit: str = "y"
    """Unit of signal values."""

    to_seconds: _t.Callable[[float], float] | float | None = None
    """Convert or scale timestamps to seconds."""

    sensor: str = ""
    """Name of the sensor."""

    def _read(self, in_path: _pl.Path) -> tuple[_bp.Signal, _pd.DataFrame]:
        df = _pd.read_csv(
            in_path,
            delimiter=self.delimiter,
            comment=self.comment,
            usecols=[self.t_column, self.y_column],
            header=0 if self.has_header else None,
            index_col=None,
            skiprows=self.skiprows,
        )

        float_timestamp = self.t_format == "float"
        if float_timestamp:
            match self.to_seconds:
                case float():
                    df[self.t_column] *= self.to_seconds
                case _ if callable(self.to_seconds):
                    df[self.t_column].apply(self.to_seconds)
                case _:
                    pass
        else:
            df[self.t_column] = _pd.to_datetime(df[self.t_column], format=self.t_format)
            df[self.t_column].apply(_u.datetime_to_seconds)

        signal = _bp.Signal(
            t=df[self.t_column].values,
            y=df[self.y_column],
            unit=self.unit,
            label=in_path.stem,
            meta=dict(sensor=self.sensor) if self.sensor else None,
        )
        return signal, df


def load_signal(path: _pl.Path) -> _bp.Signal:
    """Shorthand for loading from HDF5."""
    with h5py.File(path) as raw_f:
        signal = _bp.Signal.from_hdf(raw_f)
    return signal


def save_signal(signal: _bp.Signal, path: _pl.Path) -> None:
    """Shorthand for saving to HDF5."""
    with h5py.File(path, "w") as f:
        signal.to_hdf(f)


def download_physionet(
    *,
    db: str,
    record: str,
    channel: str,
    start: int = 0,
    stop: int | None = None,
) -> _bp.Signal:
    """Downloads a (section of) a BP signal from PhysioNet.

    :param db: database name URL part, e. g. ``autonomic-aging-cardiovascular/1.0.0``.
    :param record: record identifier, e. g. ``0001``.
    :param channel: channel name, e. g. ``NIBP``
    :param start: start index
    :param stop: stop index
    :return: a :class:`bpwave.Signal` object with metadata included.
    """
    signals, fields = wfdb.rdsamp(
        record,
        channel_names=[channel],
        sampfrom=start,
        sampto=stop,
        pn_dir=db,
    )
    return _bp.Signal(
        y=signals[:, 0],
        unit=fields["units"][0],
        fs=fields["fs"],
        label=f"{record}/{fields['sig_name'][0]}",
        meta={
            "source": "PhysioNet",
            "physionet_dir": db,
            "physionet_record": record,
            "physionet_channel": fields["sig_name"][0],
            "range": _np.array([start, stop or -1]),
        },
    )


def check_has_onsets(signal: _bp.Signal, name: str = "") -> None:
    """
    :raises ValueError: if the signal doesn't have stored onsets.
    """
    if len(signal.onsets) == 0:
        raise ValueError(f"Signal {name} must have onsets.")


def denoise(
    signal: _bp.Signal,
    *,
    f_min: float | None = None,
    f_max: float | None = 30,
    order: int = 6,
    replace: bool = False,
) -> _bp.Signal:
    """Performs forward and backward digital filtering with a Butterworth filter.

    :param signal: BP waveform
    :param f_min: lower critical frequency [Hz]
    :param f_max: higher critical frequency [Hz]
    :param order: order of the Butterworth filter
    :param replace: replace ``signal.y`` with the filtered signal, otherwise
        create a copy with the new `y`
    :return: the modified or created signal object
    """
    raw = signal.y.astype(float)
    wn: float | tuple[float, float]  # noqa
    if f_min is not None and f_max is not None:
        btype = "bandpass"
        wn = (f_min, f_max)
    elif f_max is not None:
        btype = "lowpass"
        wn = f_max
    elif f_min is not None:
        btype = "highpass"
        wn = f_min
    else:
        raise ValueError("`f_min` and `f_max` should not be both None.")

    sos = _sg.butter(order, wn, btype=btype, fs=signal.fs, output="sos")
    y = _sg.sosfiltfilt(sos, raw)
    if replace:
        signal.y = y
        return signal
    else:
        return signal.copy(y=y)


def norm01(y: _np.ndarray) -> _np.ndarray:
    """Normalizes values between 0.0 and 1.0."""
    if y.size == 0:
        return y
    range_ = y.max() - y.min()
    if range_ == 0:
        return y.astype(float) - y.min()
    return (y.astype(float) - y.min()) / range_


def invert(y: _np.ndarray) -> _np.ndarray:
    """Flips a signal wrt. amplitude."""
    if y.size == 0:
        return y
    return y.max() - y


def standardize(y: _np.ndarray) -> _np.ndarray:
    """Subtracts the mean and divides by standard deviation."""
    s = y - y.mean()
    s /= s.std()
    return s


def resample(signal: _bp.Signal, *, t: _np.ndarray) -> _bp.Signal:
    """Resamples a signal using cubic spline interpolation."""
    point_scale = len(t) / len(signal.t)
    return _bp.Signal(
        y=_ipl.CubicSpline(x=signal.t, y=signal.y)(t),
        unit=signal.unit,
        t=t,
        label=signal.label,
        meta=signal.meta,
        marks={name: values * point_scale for name, values in signal.marks.items()},
        chpoints=(
            _dc.replace(
                signal.chpoints,
                indices=[
                    _bp.CpIndices(
                        **{
                            n: int(v * point_scale)
                            for n, v in ci.without_unset().items()
                        }
                    )
                    for ci in signal.chpoints.indices
                ],
            )
            if signal.chpoints
            else None
        ),
    )


def validate_timestamps(t: _u.Array1D, check_unique: bool = True) -> None:
    """Checks if timestamps are unique and monoton increasing."""
    if check_unique:
        if len(_np.unique(t)) != len(t):
            raise ValueError("Timestamps are not unique")
    dt = _np.diff(t)
    step_back = dt < 0
    if _np.any(step_back):
        rows = _np.nonzero(step_back)[0]
        raise ValueError(f"Timestamps are not increasing at rows {rows}")
