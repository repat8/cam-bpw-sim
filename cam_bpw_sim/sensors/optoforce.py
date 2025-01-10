"""Handling measurement data of the OptoForce 3D force sensor.
"""

import dataclasses as _dc
import pathlib as _pl
import warnings

import bpwave as _bp
import numpy as _np
import pandas as _pd

from .. import utils as _u


@_dc.dataclass
class OptoForceCsvReader(_bp.SignalReader):
    """Reads OptoForce CSV output."""

    fs_override: float | None = None
    """Sampling frequency to ignore original timestamps in the CSV."""

    channel: int | None = None
    """Read a raw sensor channel {1, 2, 3, 4} instead of the length of the
    force vector.
    """

    def __post_init__(self):
        if self.fs_override is not None:
            self.validate_t = False

    def _read(self, in_path: _pl.Path) -> tuple[_bp.Signal, _pd.DataFrame]:
        data = _pd.read_csv(in_path, header=0)
        if self.channel:
            channel_name = f"cS{self.channel}"
            y = data[channel_name].to_numpy()
            bad = _np.argwhere(y == 0)
            # Zero outliers are filled from the previous value.
            # Zeros at the beginning are filled with the first nonzero value.
            if len(bad) and bad[0] == 0:
                bad = bad[1:]
                first_nonzero = y.nonzero()[0]
                y[first_nonzero:] = y[first_nonzero]
            y[bad] = y[bad - 1]
            label = f"{in_path.stem}/{channel_name}"
        else:
            y = _np.sqrt(data.X * data.X + data.Y * data.Y + data.Z * data.Z)
            label = in_path.stem
            channel_name = "vector_length"

        if self.fs_override is None:
            data["td"] = _pd.to_datetime(data["TimeStamp"], format="%H:%M:%S.%f")
            tf = data["td"].apply(_u.datetime_to_seconds).to_numpy()
            data["t"] = fix_optoforce_timestamps(tf)
            data["t_orig"] = tf
            signal_t = data["t"]
            signal_fs = None
        else:
            signal_t = None
            signal_fs = self.fs_override

        signal = _bp.Signal(
            t=signal_t,
            fs=signal_fs,
            y=y,
            unit="unit",
            label=label,
            meta=dict(
                sensor="OptoForce",
                sensor_channel=channel_name,
            ),
        )
        return signal, data


def fix_optoforce_timestamps(
    t: _u.Array1D,
    fs: float | None = None,
) -> _np.ndarray:
    """Creates a (``float``) timestamp sequence based on an original one,
    replacing obviously wrong ones with less wrong interpolation.

    :param t: original timestamp sequence
    :param fs: nominal sampling frequency, used as last resort
    :return: an array with the same length of ``t``, containing unique timestamps
    """
    res = _np.array(t, dtype=float)
    # with _v.figure(nrows=3, block=True) as (_, axes):
    #     axes[0].plot(t)
    #     axes[1].plot(_np.diff(t))
    #     axes[2].plot(_np.diff(t, n=2))
    unique_t = _np.unique(t)
    if len(unique_t) <= 1 and not fs:
        warnings.warn("Can't fix timestamps: all are the same and no fs")
        return res

    t_change_indices = _np.where(_np.diff(t, prepend=_np.nan))[0]
    for i_start, i_end in zip(t_change_indices, t_change_indices[1:]):
        if t[i_end] < t[i_start]:
            # Megkeressük az első olyan értéket, ami kisebb, mint az, ahová
            # most visszaléptünk, és onnantól interpolálunk
            t_prev = t[:i_end]
            i_start_real = t_prev[t_prev < t[i_end]].argmax()
            first_val = t_prev[i_start_real]
        else:
            i_start_real = i_start
            first_val = t[i_start]
        next_val = t[i_end]
        res[i_start_real:i_end] = _np.linspace(
            first_val, next_val, i_end - i_start_real, endpoint=False
        )

    last_change = t_change_indices[-1]
    last_known_t = t[-1]
    last_burst_len = len(t) - last_change
    # If ``fs`` is unknown, calculates based on the previous bursts
    end_t = last_burst_len / fs if fs else _np.diff(unique_t).mean()
    res[last_change:] = _np.linspace(
        last_known_t, last_known_t + end_t, last_burst_len, endpoint=False
    )
    return res
