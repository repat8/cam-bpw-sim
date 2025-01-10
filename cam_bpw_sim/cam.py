"""Generating the 3D model of the simulator's signal cam.

For nomenclature, see [DesignOfMachinery]_.

.. [DesignOfMachinery] R. Norton,
    Design of Machinery: An Introduction to the Synthesis and Analysis of Mechanisms
    and Machines. McGraw-Hill Companies, 2012, ch. 8, ISBN: 0073529354
"""

import abc
import datetime as _dt
import enum
import os
import pathlib as _pl
import shutil
import subprocess
import typing as _t
import warnings

import bpwave as _bp
import bpwave.visu as _bpv
import h5py
import matplotlib.patches as _mpa
import matplotlib.pyplot as _plt
import numpy as _np
import pydantic as _pyd
from numpy.typing import NDArray

from . import _version
from . import signal as _s
from . import utils as _u


class PreprocParams(_pyd.BaseModel):
    """Input parameters for :func:`prepare_cam_signal`."""

    from_onset: _pyd.NonNegativeInt
    """0-based index of the first detected onset to be included."""

    n_ccycles: _pyd.PositiveInt
    """Number of cardiac cycles to be included."""


class PreprocParamsOut(PreprocParams):
    """Output meatadata of :func:`prepare_cam_signal`."""

    i_start: _pyd.NonNegativeInt
    """Start index in the full signal."""

    i_stop: _pyd.PositiveInt
    """Stop index in the full signal."""


class CamInstance(_pyd.BaseModel, _u.Hdf5able):
    """Description of a single physically manufactured cam."""

    HDF_FORMAT_VERSION: _t.ClassVar[int] = 1
    _HDF_CONVERT: _t.ClassVar[_u.HdfAttrConversion] = {
        "date": lambda x: _dt.date.fromisoformat(x)
    }

    technology: str = _pyd.Field(min_length=1)
    """Production technology, e. g. ``3D_print`` or ``laser_cut``"""
    machine: str = _pyd.Field(min_length=1)
    """Exact type of the machine used for production."""
    material: str = _pyd.Field(min_length=1)
    """Exact name of the material."""
    date: _dt.date = _dt.date.today()
    """Date of production."""
    comment: str = ""
    """Additional comment."""
    seq: int = -1
    """Sequence within the same model. Automatically filled by the app."""


def prepare_cam_signal(
    full: _bp.Signal,
    params: PreprocParams,
) -> tuple[_bp.Signal, PreprocParamsOut]:
    """Transforms a signal to make it suitable for cam geeration.

    :param full: the original signal, possibly longer than desired.
        Osets must be already detected.
    :param params: preprocessing parameters
    :return:
        * the transformed signal
        * extended parameters for saving metadata
    :raises ValueError: if ``full`` doesn't have onsets
    """
    _s.check_has_onsets(full)

    i_start = full.onsets[params.from_onset]
    i_stop = full.onsets[params.from_onset + params.n_ccycles] + 1
    # ^ Include the final onset

    part = full[i_start : i_stop + 1].copy(
        label=full.label and f"{full.label}[{i_start}:{i_stop}]",
        meta={},
    )

    baseline_alg = _s.CubicSplineCorr()
    baseline_res = baseline_alg(part)
    corr = part.copy(y=baseline_res.y_corr)
    corr.meta["baseline_alg"] = baseline_alg.name
    corr.meta["baseline_version"] = _version.__version__

    with _bpv.figure(nrows=2, figsize=(20, 10)) as (_, axes):
        part.plot(
            ax=axes[0],
            legend="outside",
            title=f"onsets[{params.from_onset}] + {params.n_ccycles}",
        )
        corr.plot(ax=axes[1], legend="outside", title="Baseline corrected")
        axes[1].grid(True)

    out_data = PreprocParamsOut(**params.model_dump(), i_start=i_start, i_stop=i_stop)
    for k, v in out_data.model_dump().items():
        corr.meta[f"prep_{k}"] = v
    return corr, out_data


class Rotation(str, enum.Enum):
    """Direction of rotation."""

    cw = "cw"
    """Clockwise."""

    ccw = "ccw"
    """Counterclockwise."""


class CamParams(_pyd.BaseModel, _u.Hdf5able):
    """Cam specification parameters (without the desired pitch curve)."""

    HDF_FORMAT_VERSION: _t.ClassVar[int] = 1
    _HDF_CONVERT: _t.ClassVar[_u.HdfAttrConversion] = {
        "rotation": lambda x: Rotation(x)
    }

    name: str = _pyd.Field(min_length=1)
    """Cam/signal name."""

    amplitude: _pyd.NonNegativeFloat
    """Baseline to highest peak amplitude of the pitch curve."""

    r: _pyd.NonNegativeFloat
    """Baseline radius of the cam."""

    r_rim: _pyd.NonNegativeFloat
    """Radius of the quality checker rim. ``r_rim < r`` must hold."""

    r_follower: _pyd.NonNegativeFloat
    """Radius of the cam follower."""

    d_shaft: _pyd.NonNegativeFloat
    """Diameter of the central shaft
    (with production tolerance taken into account)."""

    rotation: Rotation = Rotation.ccw
    """Direction of rotation."""

    invert: bool = False
    """Invert pitch curve, i. e. rising edges of the signal will be
    realized as decreasing radius."""

    resample: bool = False
    """Resample pitch curve [not yet implemented]."""


class CamData:
    def __init__(self, signal: _bp.Signal, params: CamParams):
        """Generates the coordinates for the cam surface, the baseline rim and the
        camshaft based on a precomputed pitch curve."""
        assert params.r_rim < params.r

        self._orig_signal = signal
        self._params = params
        self._nominal_signal = self._calc_nominal_signal()
        self._pitch_curve = self._calc_pitch_curve()

        self._theta = _np.linspace(0, 2 * _np.pi, len(self._pitch_curve))
        self._pitch_curve_coords = self._calc_pitch_curve_coords()
        self._cam_profile_coords = self._calc_cam_profile_coords()

        self._anomalies = self._check_quality()

    @property
    def orig_signal(self) -> _bp.Signal:
        """The original input signal in original scale."""
        return self._orig_signal

    @property
    def nominal_signal(self) -> _u.NDArray1D[_np.float64]:
        """Signal scaled to the desired amplitude (but without inversion)."""
        return self._nominal_signal

    @property
    def params(self) -> CamParams:
        """Cam specification parameters."""
        return self._params

    @property
    def theta(self) -> _u.NDArray1D[_np.float64]:
        """Angle vector for cam profile data points."""
        return self._theta

    @property
    def pitch_curve(self) -> _u.NDArray1D[_np.float64]:
        """Pitch curve, inversion and rotation direction applied."""
        return self._pitch_curve

    @property
    def pitch_curve_coords(self) -> _u.NDArray2D[_np.float64]:
        """(x, y) coordinates of the pitch curve."""
        return self._pitch_curve_coords

    @property
    def cam_profile_coords(self) -> _u.NDArray2D[_np.float64]:
        r"""(x, y) coordinates of the cam profile.

        Given the pitch curve coordinates :math:`K: (x, y)`,
        calculated as adding signal amplitude :math:`S(\theta)` to
        cam baseline radius ``params.r``, the coordinates of the cam profile
        are computed as [IntrMech]_

        .. math::
            \left\{ \begin{aligned}
            x_P &= x - r \cdot
            \frac{\text{d}y/\text{d}\theta}
            {\sqrt{(\text{d}x/\text{d}\theta)^2 + (\text{d}y/\text{d}\theta)^2}} \\
            y_P &= y + r \cdot
            \frac{\text{d}x/\text{d}\theta}
            {\sqrt{(\text{d}x/\text{d}\theta)^2 + (\text{d}y/\text{d}\theta)^2}}
            \end{aligned} \right.

        .. [IntrMech] https://www.cs.cmu.edu/~rapidproto/mechanisms/chpt6.html
        """
        return self._cam_profile_coords

    @property
    def anomalies(self) -> _u.NDArray1D[_np.int64]:
        """Indices of cam profile anomalies, such as undercutting.
        Can be used to index :attr:`theta`, :attr:`cam_profile_coords` etc."""
        return self._anomalies

    def plot_nominal_signal(self, ax: _plt.Axes) -> None:
        """Plots the nominal signal represented by the cam, in linear form."""
        ax.hlines(
            [self._nominal_signal.min(), self._nominal_signal.max()],
            xmin=0,
            xmax=360,
            colors="0.5",
            lw=0.5,
        )
        ax.plot(_np.rad2deg(self._theta), self._nominal_signal, "k", lw=0.5)
        ax.grid(True)
        ax.set(
            xlabel=r"$\omega$ [°]",
            ylabel="$y$ [mm]",
            xlim=[0, 360],
        )

    def plot_cam(self, ax: _plt.Axes) -> None:
        """Plots a drawing of the cam."""
        for x, y in self._cam_profile_coords[self._anomalies]:
            ax.plot(x, y, "r+")
        ax.plot(
            self._pitch_curve_coords[:, 0],
            self._pitch_curve_coords[:, 1],
            "0.5",
            lw=0.5,
        )
        ax.plot(
            self._cam_profile_coords[:, 0], self._cam_profile_coords[:, 1], "k", lw=0.5
        )
        ax.plot(0, 0, "k+")
        ax.add_patch(
            _mpa.Circle((0, 0), radius=self._params.r, color="0.7", lw=0.5, fill=False)
        )
        ax.add_patch(
            _mpa.Circle(
                (0, 0), radius=self._params.d_shaft / 2, color="k", lw=0.5, fill=False
            )
        )
        ax.set(aspect="equal")

    def save(self, path: str | os.PathLike, *, force: bool = False):
        """Saves curves and parameters to a HDF5 file."""
        with h5py.File(path, "w" if force else "w-") as f:
            f.attrs["version"] = _version.__version__
            f.attrs["created"] = _dt.datetime.now().isoformat()

            params_ds = f.create_dataset("params", dtype=h5py.Empty("f"))
            for name, value in self.params.model_dump(mode="json").items():
                params_ds.attrs[name] = value

            self.orig_signal.to_hdf(f.create_group("orig_signal"))

            f.create_dataset("theta", data=self.theta)
            f.create_dataset("nominal_signal", data=self.nominal_signal)
            f.create_dataset("pitch_curve", data=self.pitch_curve)
            f.create_dataset("pitch_curve_coords", data=self.pitch_curve_coords)
            f.create_dataset("cam_profile_coords", data=self.cam_profile_coords)

    def _calc_nominal_signal(self) -> NDArray[_np.float64]:
        sig = self._orig_signal  # TODO resample
        nom = _s.norm01(sig.y) * self._params.amplitude
        return nom

    def _calc_pitch_curve(self) -> NDArray[_np.float64]:
        curve = self._nominal_signal
        if self._params.invert:
            curve = curve.max() - curve
        if self._params.rotation == Rotation.ccw:
            curve = curve[::-1]
        return curve

    def _calc_pitch_curve_coords(self) -> NDArray[_np.float64]:
        curve = self._pitch_curve
        unit_circle = _np.c_[_np.cos(self._theta), _np.sin(self._theta)]
        incr_radii = curve + self._params.r + self._params.r_follower
        return _np.c_[incr_radii, incr_radii] * unit_circle

    def _calc_cam_profile_coords(self) -> NDArray[_np.float64]:
        if self._params.r_follower == 0:
            return self._pitch_curve_coords
        else:
            # https://www.cs.cmu.edu/~rapidproto/mechanisms/chpt6.html
            xk = self._pitch_curve_coords[:, 0]
            yk = self._pitch_curve_coords[:, 1]
            dxk = _np.diff(xk)
            dyk = _np.diff(yk)
            denom = _np.sqrt(dxk * dxk + dyk * dyk)
            xp = xk[:-1] - self._params.r_follower * dyk / denom
            yp = yk[:-1] + self._params.r_follower * dxk / denom
            return _np.c_[xp, yp]

    def _check_quality(self):
        # Detecting undercutting:
        # Transform back to Cartesian coordinate system and check for backsteps
        x = self._cam_profile_coords[:, 0]
        y = self._cam_profile_coords[:, 1]
        check_theta = _np.arctan2(y, x)
        d_theta = _np.diff(check_theta)
        backsteps = _np.nonzero((-_np.pi < d_theta) & (d_theta < 0.0))[0]
        if len(backsteps):
            warnings.warn("Undercutting occurs, please check `.anomalies`")
        return backsteps


class CamStlGenerator(abc.ABC):
    def __init__(self, cam: CamData, cams_folder: str | os.PathLike):
        """Generates the STL model from the precomputed cam coordinates.

        :param cam: cam parameters and coordinates object
        :param cams_folder: path to the parent of the output folder.
            A subfolder will be automatically created to ensure consistent naming.
            If doesn't exist, it will be created;
            if exists, contents of the specific subfolder will be overwritten.
        """
        self._cam = cam
        self._folder = (
            _pl.Path(cams_folder)
            / self._cam.params.name
            / self.model_label
            / self.version
        )
        self._name_base = f"{self._cam.params.name}_{self.model_label}"
        self._overwriting = self._folder.exists()

        self._folder.mkdir(parents=True, exist_ok=True)

    @property
    @abc.abstractmethod
    def version(self) -> str:
        """Model version. Will be a part of file path."""
        pass

    @property
    def model_label(self) -> str:
        """An identifier depending on parameters. Will be a part of file path."""
        par = self._cam.params
        return (
            f"d{par.d_shaft * 100:g}r{par.r * 100:g}a{par.amplitude * 100:g}"
            f"rf{par.r_follower * 100:g}"
            f"{'i' if par.invert else ''}{'c' if par.rotation == Rotation.cw else ''}"
        )

    @property
    def folder(self) -> _pl.Path:
        """The folder created to contain cam model and data."""
        return self._folder

    def _save_metadata(self, force: bool) -> None:
        self._cam.save(
            self._folder / f"{self._name_base}.cam.hdf5",
            force=force,
        )
        with (self._folder / f"{self._name_base}.cam.json").open("w") as f:
            print(self._cam.params.model_dump_json(indent=2), file=f)

    def _plot(self):
        fig, ax = _plt.subplots(figsize=(15, 5))
        self._cam.plot_nominal_signal(ax)
        fig.tight_layout()
        fig.savefig(self._folder / f"{self._name_base}_nom.png")
        fig, ax = _plt.subplots(figsize=(10, 10))
        self._cam.plot_cam(ax)
        fig.tight_layout()
        fig.savefig(self._folder / f"{self._name_base}.cam.png")

    @abc.abstractmethod
    def _generate_stl(self) -> _pl.Path:
        """Generate the actual STL model based on the precomputed data."""
        pass

    def run(self, *, force: bool = False) -> _pl.Path | None:
        """Run the STL generation process.
        It calls the specialized method :meth:`_generate_stl`.

        :param force: override files if they already exist.
        :return: the path of the STL file if it was created.
        """
        if any(self._folder.iterdir()) and not force:
            warnings.warn(f"{self._folder} is not empty. Try again with `force=True`")
            return None
        self._plot()
        self._save_metadata(force)
        path = self._generate_stl()
        return path


class OpenSCADCamGenerator(CamStlGenerator):
    """Generates STL using OpenSCAD."""

    @property
    def version(self):
        return "SC.1.1"

    def _generate_stl(self) -> _pl.Path:
        stl_path = self._folder / f"{self._cam.params.name}_{self.model_label}.cam.stl"
        scad_path = self._folder / f"{self._name_base}.cam.scad"
        shutil.copy(_pl.Path(__file__).parent / "cams.scad", self._folder)

        coords = _np.vstack(
            [
                self._cam.cam_profile_coords,
                self._cam.cam_profile_coords[-1],
            ]
        ).tolist()
        with scad_path.open("w") as scad:
            print("include <cams.scad>", file=scad)
            print(
                f"""
                bp_simulator_cam(
                    $fn=60,
                    signal_points={coords},
                    r={self._cam.params.r},
                    r_rim={self._cam.params.r_rim},
                    r_follower={self._cam.params.r_follower},
                    name="{self._cam.params.name}",
                    amplitude={self._cam.params.amplitude},
                    d_shaft={self._cam.params.d_shaft},
                    rotation="{self._cam.params.rotation.value}",
                    cam_version="{self.version}"
                );
                """,
                file=scad,
            )
        subprocess.run(
            ["openscad", "-o", str(stl_path), str(scad_path)],
            check=True,
        )
        return stl_path


def curve_on_circle(
    y: NDArray[_np.float64],
    theta: NDArray[_np.float64],
) -> NDArray[_np.float64]:
    unit_circle = _np.c_[_np.cos(theta), _np.sin(theta)]
    return _np.c_[y, y] * unit_circle
