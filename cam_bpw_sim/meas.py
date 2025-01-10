"""Measurement data processing."""

import dataclasses as _dc
import pathlib as _pl
import typing as _t

import bpwave as _bp
import h5py
import matplotlib.markers as _mpm
import matplotlib.patches as _mpp
import matplotlib.pyplot as _plt
import numpy as _np
import pydantic as _pyd

from . import cam as _c
from . import utils as _u


class SimulatorParams(_pyd.BaseModel, _u.Hdf5able):
    """Simulator setup."""

    HDF_FORMAT_VERSION: _t.ClassVar[int] = 1
    _HDF_SKIP: _t.ClassVar[set[str]] = {"cams"}

    label: str
    """Identifier of the simulator setup."""

    lever_class: int
    """``1``: the fulcrum is between the cam and the sensor;
    ``2``: the sensor and the cam are on the same side of the fulcrum.
    """

    lever_r1: _pyd.PositiveFloat
    """The distance between the fulcrum and the cam follower."""

    lever_r2: _pyd.PositiveFloat
    """The distance between the fulcrum and the vertical lever follower."""

    fulcrum_x: _pyd.PositiveFloat
    """Horizotal distance between the fulcrum and the center of the cam."""

    cam_follower_d: _pyd.PositiveFloat
    """Diameter of the cam follower
    (in case of mushroom follower type, radius × 2).
    """

    cam_follower_h: _pyd.NonNegativeFloat
    """Distance between the contact point of the cam follower
    and the center of the fulcrum shaft.
    """

    lever_follower_d: _pyd.NonNegativeFloat
    """Diameter of the rolling head of the vertical lever follower."""

    cams: dict[str, str] = {}
    """Mapping of cam names to actual file paths of cam instances used,
    relative to the ``cam`` subfolder of the apllication home folder.

    E. g.:

    .. code-block:: json

        "cams": {
            "AAC4": "AAC4_0/d605r3000a100rf0/SC.1.0/inst/1"
        }
    """

    @property
    def eccentricity(self) -> float:
        """The eccentricity of the cam, ie. the signed distance between the
        trajectory of the cam follower and the center of the cam.
        """
        return self.fulcrum_x - self.lever_r1

    def draw(self, *, r_cam: float = 30.0) -> _plt.Axes:
        """Visualizes the current parameters.

        :param r_cam: the baseline radius of the cam.
        """
        x_lever_head = self.eccentricity
        y_lever_head = 40 + r_cam + self.cam_follower_d / 2
        h_fulcrum = y_lever_head + self.cam_follower_h
        x_lever_follower = (
            self.fulcrum_x + self.lever_r2 * {1: 1, 2: -1}[self.lever_class]
        )
        y_lever_follower = h_fulcrum + self.lever_follower_d / 2

        # Structure

        fig, ax = _plt.subplots(
            figsize=(int((self.fulcrum_x + 100) / (y_lever_follower + 90) * 5), 5)
        )
        ax.set(
            aspect="equal",
            title=self.label,
            xlabel="[mm]",
            ylabel="[mm]",
        )

        # Base
        ax.plot([-r_cam, self.fulcrum_x + 50], [0, 0], "k")
        # Cam holder
        ax.add_patch(
            _mpp.Polygon([[-10, 0], [10, 0], [0, 40]], closed=True, fill=False)
        )
        # Cam
        ax.add_patch(_mpp.Circle((0, 40), r_cam, fill=False))
        # Cam follower
        ax.add_patch(
            _mpp.Circle(
                (x_lever_head, y_lever_head), self.cam_follower_d / 2, fill=False
            )
        )
        # Fulcrum
        ax.add_patch(
            _mpp.Polygon(
                [
                    [self.fulcrum_x - 10, 0],
                    [self.fulcrum_x + 10, 0],
                    [self.fulcrum_x, h_fulcrum],
                ],
                closed=True,
                fill=False,
            )
        )
        # Lever
        ax.plot(
            [x_lever_head, x_lever_head, max(x_lever_follower, self.fulcrum_x) + 40],
            [y_lever_head, h_fulcrum, h_fulcrum],
            "k",
        )
        # Lever follower
        ax.add_patch(
            _mpp.Circle(
                (x_lever_follower, y_lever_follower),
                self.lever_follower_d / 2,
                fill=False,
            )
        )
        # Sensor holder
        ax.add_patch(
            _mpp.Polygon(
                [
                    [x_lever_follower, y_lever_follower],
                    [x_lever_follower - 10, y_lever_follower + 60],
                    [x_lever_follower + 10, y_lever_follower + 60],
                ],
                closed=True,
                fill=False,
            )
        )

        # Dimensions

        aux_props = dict(lw=0.5, color="k")
        levels = _np.linspace(0, -20, 4)
        ax.plot([x_lever_head, x_lever_head], [levels[-1], y_lever_head], **aux_props)
        ax.plot([0, 0], [levels[-1], 40], **aux_props)
        ax.plot(
            [x_lever_follower, x_lever_follower],
            [levels[-1], y_lever_follower],
            **aux_props,
        )
        ax.plot([self.fulcrum_x, self.fulcrum_x], [levels[-1], h_fulcrum], **aux_props)

        def dim_line(ax, x1, x2, y, text, *, above=False):
            props = dict(lw=0.5, color="k")
            text_props = (
                dict(y=y + 3, va="bottom") if above else dict(y=y - 3, va="top")
            )
            x_min = min(x1, x2)
            x_max = max(x1, x2)
            if x_max - x_min < 50:
                ax.plot([x_min - 5, x_max + 20], [y, y], **props)
                ax.plot(x_min, y, "k", marker=_mpm.CARETRIGHT)
                ax.plot(x_max, y, "k", marker=_mpm.CARETLEFT)
                ax.text(x_max + 10, s=text, **text_props)
            else:
                ax.plot([x_min, x_max], [y, y], **props)
                ax.plot(x_min, y, "k", marker=_mpm.CARETLEFT)
                ax.plot(x_max, y, "k", marker=_mpm.CARETRIGHT)
                ax.text((x_min + x_max) / 2, s=text, ha="center", **text_props)

        dim_line(ax, x_lever_head, 0, levels[2], f"$\\epsilon = {x_lever_head}$")
        dim_line(
            ax,
            x_lever_follower,
            self.fulcrum_x,
            levels[2],
            f"$r_2 = {self.lever_r2}$",
        )
        dim_line(
            ax, x_lever_head, self.fulcrum_x, levels[1], f"$r_1 = {self.lever_r1}$"
        )
        dim_line(
            ax,
            x_lever_follower - self.lever_follower_d / 2,
            x_lever_follower + self.lever_follower_d / 2,
            y_lever_follower,
            f"$d_{{lf}} = {self.lever_follower_d}$",
            above=True,
        )
        dim_line(
            ax,
            x_lever_head - self.cam_follower_d / 2,
            x_lever_head + self.cam_follower_d / 2,
            y_lever_head,
            f"$d_f = {self.cam_follower_d}$",
            above=True,
        )

        return ax


class MeasParams(_pyd.BaseModel, _u.Hdf5able):
    """Parameters of a single measurement."""

    protocol: str
    """Measurement protocol identifier."""

    cam_inst: str
    """Cam instance key from ``simulator_setup.json``."""

    u: float
    """Voltage"""


@_dc.dataclass(kw_only=True)
class MeasEnvironment:
    """The complete description of the measurement environment,
    including the cam and simulator parameters.
    """

    nominal: _bp.Signal
    """Preprocessed nominal signal used for generating the cam."""

    cam_params: _c.CamParams
    """Parameters of generating the cam."""

    cam_inst: _c.CamInstance
    """Parameters of the specific cam instance used."""

    simulator: SimulatorParams
    """Parameters of the simulator setup."""

    cam_params_source: _pl.Path
    """Original path of the cam parameters file,
    relative to the application home folder.
    """

    cam_inst_source: _pl.Path
    """Original path of the cam instance parameters file,
    relative to the application home folder.
    """


@_dc.dataclass(kw_only=True)
class MeasWithMeta(MeasEnvironment, _u.Hdf5able):
    """The measured signal together with all parameters.

    This object is intended to be serialized to the HDF5 measurement files.
    """

    HDF_FORMAT_VERSION: _t.ClassVar[int] = 1

    measured: _bp.Signal
    """Measured signal."""

    meas_params: MeasParams
    """All parameters of the measurement environment."""

    def to_hdf(self, root: h5py.Group) -> None:
        super().type_info_to_hdf(root)
        self.measured.to_hdf(root.create_group("measured"))
        self.nominal.to_hdf(root.create_group("nominal"))
        self.cam_params.to_hdf(cam_params_gr := root.create_group("cam_params"))
        self.cam_inst.to_hdf(cam_inst_gr := root.create_group("cam_inst"))
        self.simulator.to_hdf(root.create_group("simulator"))
        self.meas_params.to_hdf(root.create_group("meas_params"))
        cam_params_gr.attrs["source_file"] = str(self.cam_params_source)
        cam_inst_gr.attrs["source_file"] = str(self.cam_inst_source)
