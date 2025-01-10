import bpwave as _bp
import matplotlib.pyplot as _plt
import numpy as _np

from . import cam as _c
from . import signal as _s
from . import utils as _u


def plot_cam_vs_measured(
    nominal_fcr: _bp.Signal,
    measured_fcr: _u.NDArray1D[_np.float64],
    cam_params: _c.CamParams,
) -> tuple[_plt.Figure, _plt.Axes, _plt.Axes]:
    """Plots the nominal cam profile together with the measured one
    and with their difference.

    :param nominal_fcr: nominal signal (with matched ``fs``).
    :param measured_fcr: measured (e. g. averaged) full cam rotation.
    :param cam_params: parameters used at cam generation.
    :return: figure and 2 subplots
    """
    assert len(nominal_fcr.y) == len(measured_fcr)
    nominal_scaled = _s.norm01(nominal_fcr.y) * cam_params.amplitude
    measured_scaled = _s.norm01(measured_fcr) * cam_params.amplitude
    fig = _plt.figure(figsize=(10, 12))
    gr = fig.add_gridspec(nrows=3)
    ax = fig.add_subplot(gr[:2], projection="polar")
    ax.plot(
        _np.linspace(0, 2 * _np.pi, num=len(measured_fcr)),
        cam_params.r + measured_scaled,
        lw=0.5,
    )
    ax.plot(
        _np.linspace(0, 2 * _np.pi, num=len(nominal_fcr.y)),
        cam_params.r + nominal_scaled,
        lw=0.5,
    )
    ax.plot(
        _np.linspace(0, 2 * _np.pi, num=len(measured_fcr)),
        cam_params.r * 0.75
        + _s.norm01(_np.abs(nominal_scaled - measured_scaled)) * cam_params.r * 0.25,
        lw=0.5,
        color="r",
    )
    ax.set(
        rmax=cam_params.r + 2 * cam_params.amplitude,
        rticks=[cam_params.r * 0.75, cam_params.r, cam_params.r + cam_params.amplitude],
    )
    ax.grid(True)
    ax2 = fig.add_subplot(gr[-1])
    theta = _np.linspace(0, 360, num=len(measured_fcr))
    ax2.step(theta, nominal_scaled, where="mid", label="nom.", lw=0.5)
    ax2.step(theta, measured_scaled, where="mid", label="meas.", lw=0.5)
    ax2.fill_between(
        theta, nominal_scaled, measured_scaled, color="r", alpha=0.5, step="mid"
    )
    ax2.set(xlim=theta[[0, -1]])
    ax2.grid(True)
    ax2.legend()
    return fig, ax, ax2
