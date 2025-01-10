import abc
import collections.abc as _ca
import dataclasses as _dc
import datetime as _dt
import enum
import os
import pathlib as _pl
import typing as _t

import h5py
import matplotlib.pyplot as _plt
import numpy as _np
import numpy.typing as _npt
import pydantic as _pyd
import typing_extensions as _te

from . import _version

_T = _t.TypeVar("_T")
PathLike: _t.TypeAlias = str | os.PathLike
Array1D: _t.TypeAlias = _ca.Sequence[_T] | _np.ndarray
Array2D: _t.TypeAlias = _ca.Sequence[_ca.Sequence[_T]] | _np.ndarray
NDArray1D: _t.TypeAlias = _t.Annotated[_npt.NDArray[_T], "ndim==1"]
NDArray2D: _t.TypeAlias = _t.Annotated[_npt.NDArray[_T], "ndim==2"]
NDArray3D: _t.TypeAlias = _t.Annotated[_npt.NDArray[_T], "ndim==3"]
HdfAttrConversion: _t.TypeAlias = dict[str, _t.Callable[[_t.Any], _t.Any]]


class Hdf5able(abc.ABC):
    HDF_FORMAT_VERSION: _t.ClassVar[int] = 0
    _HDF_SKIP: _t.ClassVar[set[str]] = set()
    _HDF_CONVERT: _t.ClassVar[HdfAttrConversion] = {}

    def to_hdf(self, root: h5py.Group) -> None:
        skip = self._HDF_SKIP
        obj_to_hdf(self, root, skip=skip)
        self.type_info_to_hdf(root)

    def type_info_to_hdf(self, root: h5py.Group) -> None:
        root.attrs["_type"] = f"{self.__module__}.{type(self).__name__}"
        root.attrs["_package_version"] = _version.__version__
        root.attrs["_format_version"] = self.HDF_FORMAT_VERSION

    @classmethod
    def from_hdf(cls, root: h5py.Group) -> _te.Self:
        if (t := root.attrs["_type"]) != f"{cls.__module__}.{cls.__name__}":
            raise ValueError(f"HDF5 file has incompatible type {t}")
        if (v := root.attrs["_format_version"]) != cls.HDF_FORMAT_VERSION:
            raise ValueError(f"HDF5 file has incompatible format version {v}")
        conversion = cls._HDF_CONVERT
        kwargs = {
            key: c(value) if (c := conversion.get(key)) else value
            for key, value in root.attrs.items()
            if not key.startswith("_")
        }
        return cls(**kwargs)  # type: ignore


def obj_to_hdf(obj: _t.Any, root: h5py.Group, *, skip: set | None = None) -> None:
    match obj:
        case _pyd.BaseModel():
            obj_to_hdf(obj.model_dump(), root, skip=skip)
        case _ if hasattr(obj, "to_hdf"):
            obj.to_hdf(root)
        case _ if _dc.is_dataclass(obj):
            obj_to_hdf(obj.__dict__, root, skip=skip)
        case dict():
            for key in obj.keys() - (skip or set()):
                value_to_hdf(key, obj[key], root)
        case _:
            raise ValueError(f"obj_to_hdf: cannot handle {type(obj)}")


def value_to_hdf(key: str, value: _t.Any, root: h5py.Group) -> None:
    v_conv: _t.Any
    match value:
        case _dt.date() | _dt.datetime():
            v_conv = value.isoformat()
        case set():
            v_conv = list(value)
        case _pl.Path():
            v_conv = str(value)
        case enum.Enum():
            v_conv = value.value
        case list() | _np.ndarray() | int() | float() | str():
            v_conv = value
        case _:
            raise ValueError(f"value_to_hdf: cannot handle {type(value)} at {key!r}")
    root.attrs[key] = v_conv


def convert_path(path: PathLike, create: bool) -> _pl.Path:
    """Converts ``PathLike`` to ``Path`` and creates parent folders if requested."""
    p = _pl.Path(path)
    if create:
        p.parent.mkdir(parents=True, exist_ok=True)
    else:
        if not p.exists():
            raise FileNotFoundError(path)
    return p.resolve()


def datetime_to_seconds(t: _dt.datetime) -> float:
    return t.microsecond / 1e6 + t.second + t.minute * 60 + t.hour * 3600


def ensure_ax(ax: _plt.Axes | None) -> _plt.Axes:
    if not ax:
        _, ax = _plt.subplots()
    assert ax  # Make mypy happy
    return ax


def mirror_to_diag(m: _np.ndarray) -> None:
    """Mirrors lower triangle matrix to upper triangle."""
    m += m.T
    m -= _np.diag(_np.diag(m))
