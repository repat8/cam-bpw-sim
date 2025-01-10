"""Higher level app functionality to be used in the user interfaces.

.. _Autonomic Aging: https://www.physionet.org/content/autonomic-aging-cardiovascular/1.0.0/
"""

import dataclasses as _dc
import glob
import json
import pathlib as _pl
import re
import typing as _t

import bpwave as _bp
import bpwave.visu as _bpv
import h5py
import matplotlib.pyplot as _plt
import papermill
import pydantic as _pyd
import tomlkit
import typer as _ty

from . import _version
from . import cam as _c
from . import meas as _m
from . import signal as _s

APP_NAME = "cam-bpw-sim"
CONFIG_FOLDER_PATH = _pl.Path(_ty.get_app_dir(APP_NAME))
CONFIG_FILE_PATH = CONFIG_FOLDER_PATH / "config.toml"


class WorkflowError(RuntimeError):
    """Error resulting from inconsistent command usage."""


class NotConfirmed(RuntimeError):
    pass


class CamInstanceConfig(_pyd.BaseModel):
    """Choices for cam instance registration."""

    technology: list[str] = []
    machine: list[str] = []
    material: list[str] = []


class AppConfig(_pyd.BaseModel):
    """Top level app configuration."""

    simulator_home: _pyd.DirectoryPath
    """Path of the folder to be used by the app."""

    cam_instance: CamInstanceConfig
    """Choices for cam instance registration."""

    @property
    def raw_signal_folder(self) -> _pl.Path:
        return self.simulator_home / "raw_signal"

    @property
    def prep_signal_folder(self) -> _pl.Path:
        return self.simulator_home / "prep_signal"

    @property
    def cam_folder(self) -> _pl.Path:
        return self.simulator_home / "cam"

    @property
    def measurement_folder(self) -> _pl.Path:
        return self.simulator_home / "meas"

    def relative(self, p: _pl.Path) -> _pl.Path:
        """Path relative to :attr:`simulator_home`."""
        return p.relative_to(self.simulator_home)


def init_app_config(
    *,
    on_exists: _t.Callable[[_pl.Path], None],
    on_create: _t.Callable[[_pl.Path], None],
) -> _pl.Path:
    """Creates the config folder and the config file, if it doesn't already exist.

    :param on_exists: called if the config file already exists
    :param on_create: called before creating the file
    :returns: the path of the config file
    """
    CONFIG_FOLDER_PATH.mkdir(parents=True)
    if CONFIG_FILE_PATH.exists():
        on_exists(CONFIG_FILE_PATH)
    else:
        on_create(CONFIG_FILE_PATH)
        with CONFIG_FILE_PATH.open("w") as f:
            for line in [
                'simulator_home = "EDIT ME"',
                "[cam_instance]",
                "technology = []",
                "machine = []",
                "material = []",
            ]:
                print(line, file=f)
    return CONFIG_FILE_PATH


def read_config() -> AppConfig:
    """Loads the config file content.

    :returns: :class:`AppConfig`
    :raises WorkflowError: if the config file is missing.
    """
    if not CONFIG_FILE_PATH.exists():
        raise WorkflowError("app.config.missing")
    with CONFIG_FILE_PATH.open() as cfg:
        return AppConfig(**tomlkit.load(cfg).value)


def init_simulator_home(
    *,
    on_home: _t.Callable[[_pl.Path], None],
    on_create: _t.Callable[[_pl.Path], None],
) -> _pl.Path:
    """Creates the config folder and the config file, if it doesn't already exist.

    :param on_home: called before checking home content and initialization.
    :param on_create: called after creating each subfolders.
    :returns: the path of the app home folder.
    :raises WorkflowError: if the app home is not empty (already initialized)
    """
    config = read_config()
    home = config.simulator_home
    on_home(home)
    if list(home.iterdir()):
        raise WorkflowError("app.simulator-home.not-empty")
    for p in [
        config.raw_signal_folder,
        config.prep_signal_folder,
        config.cam_folder,
        config.measurement_folder,
    ]:
        p.mkdir(exist_ok=False)
        on_create(p)
    return home


def download_physionet_autonomic_aging(
    *,
    record: str,
    start: int = 0,
    stop: int | None = None,
    force: bool = False,
) -> tuple[_pl.Path, _bp.Signal]:
    """Downloads a signal (or a part of it) from the `Autonomic Aging`_ dataset
    of PysioNet.

    :param record: record identifier, e. g. ``0001``.
    :param start: start index
    :param stop: stop index
    :param force: overwrite existing downloaded file.
    :return: the file path and the sigal object
    """
    config = read_config()
    range_ = "full" if not start and not stop else f"{start}_{stop or 'end'}"
    fname = f"physionet_aac_{record}_{range_}.hdf5"
    fpath = config.raw_signal_folder / fname
    if fpath.exists() and not force:
        raise WorkflowError("app.signal.already-exists")
    wf = _s.download_physionet(
        db="autonomic-aging-cardiovascular/1.0.0",
        record=record,
        channel="NIBP",
        start=start,
        stop=stop,
    )
    with h5py.File(fpath, "w") as f:
        wf.to_hdf(f)
    return fpath, wf


def list_raw_signals() -> list[_pl.Path]:
    """Collects raw signal file paths."""
    config = read_config()
    return [
        p.relative_to(config.simulator_home) for p in config.raw_signal_folder.iterdir()
    ]


def list_prep_signals() -> list[_pl.Path]:
    """Collects preprocessed signal file paths."""
    config = read_config()
    return [
        p.relative_to(config.simulator_home)
        for p in config.prep_signal_folder.iterdir()
    ]


def view_signal(rel_path: _pl.Path) -> _bp.Signal:
    """Plots a signal.

    :param rel_path: path relative to the app home.
    """
    config = read_config()
    signal = _s.load_signal(config.simulator_home / rel_path)
    with _bpv.figure(figsize=(20, 5)) as (_, axes):
        signal.plot(ax=axes[0], title=str(rel_path))
        axes[0].grid(True)
    return signal


def prep_detect_chpoints(
    name: str,
    *,
    force: bool = False,
) -> tuple[_pl.Path, _bp.Signal]:
    """Creates a copy of the signal and annotates it with the characteristic
    points.

    The signal will be denoised with a lowpass filter.

    :param name: file name of the raw signal.
    :param force: overwrite existing file.
    :return: relative path of the signal and the signal object.
    """
    name_p = _pl.Path(name)
    config = read_config()
    in_path = config.raw_signal_folder / name
    raw = _s.load_signal(in_path)

    f_max = 30.0
    order = 6
    signal = _s.denoise(raw, f_max=f_max, order=order)
    signal.label = f"denoised {raw.label}"
    signal.meta = {
        "base": str(config.relative(in_path)),
        "denoise_alg": f"{_s.denoise.__module__}.{_s.denoise.__name__}",
        "denoise_version": _version.__version__,
        "denoise_params": json.dumps(dict(order=order, f_max=f_max)),
    }

    p_res = _s.ScipyFindPeaks()(signal)
    signal.chpoints = p_res.chpoints

    with _bpv.figure(
        title=name, figsize=(20, 10), nrows=2, sharex=True, sharey=True
    ) as (_, axes):
        raw.plot(ax=axes[0])
        signal.plot(ax=axes[0], points=False, onsets=False, append=True)
        signal.plot(ax=axes[1])

    out_path = config.prep_signal_folder / name_p.with_stem(f"{name_p.stem}.chpoints")
    if out_path.exists() and not force:
        raise WorkflowError("app.signal.already-exists")
    _s.save_signal(signal, out_path)

    return out_path.relative_to(config.simulator_home), signal


def add_cam_signal(
    signal: _bp.Signal,
    cam_signal_name: str,
    *,
    force: bool = False,
) -> _pl.Path:
    """Saves a cam signal.

    It is assumed that the signal has already been preprocessed and ready for
    cam generation.

    :param signal: preprocessed signal.
    :param cam_signal_name: identifier for the preprocessed signal section
        to be creted now.
    :param force: overwrite file if exists.
    :raises WorkflowError: if the file exists and not forced.
    """
    config = read_config()
    out_path = config.prep_signal_folder / f"{cam_signal_name}.camsig.hdf5"
    if out_path.exists() and not force:
        raise WorkflowError("app.signal.already-exists")
    _s.save_signal(signal, out_path)
    return out_path.relative_to(config.simulator_home)


def prep_for_cam(
    full_signal_name: str,
    cam_signal_name: str,
    *,
    from_onset: int,
    n_ccycles: int,
    force: bool = False,
) -> tuple[_pl.Path, _bp.Signal]:
    """Performs the default preprocessing for cam generation.

    :param full_signal_name: preprocessed signal file name.
    :param cam_signal_name: identifier for the preprocessed signal section
        to be creted now.
    :param from_onset: the index of the first cardiac cycle to be included.
    :param n_ccycles: number of cardiac cycles to be included.
    :param force: overwrite file if exists.
    :raises WorkflowError: if the file exists and not forced.
    """
    config = read_config()
    in_path = config.prep_signal_folder / full_signal_name
    with_onsets = _s.load_signal(in_path)
    cam_signal, _ = _c.prepare_cam_signal(
        with_onsets, _c.PreprocParams(from_onset=from_onset, n_ccycles=n_ccycles)
    )
    cam_signal.meta["base"] = str(config.relative(in_path))
    out_path = add_cam_signal(cam_signal, cam_signal_name, force=force)
    return out_path, cam_signal


def generate_cam(
    params: _c.CamParams,
    *,
    ask_continue: _t.Callable[[], bool],
    force: bool = False,
    on_stl_started: _t.Callable[[], None],
    on_stl_finished: _t.Callable[[], None],
) -> _pl.Path | None:
    """Generates the printable STL model of the cam based on ``params``.

    :param params: cam parameters.
    :param ask_continue: confirmation callback before generating the model.
    :param force: overwrite model, if already exists.
    :param on_stl_started: called before generating STL.
    :param on_stl_finished: called when STL is ready.
    :return: the path of the STL file (or ``None`` if aborted or failed).
    :raises WorkflowError: if the cam already has instances
    """
    config = read_config()
    signal = _s.load_signal(config.prep_signal_folder / f"{params.name}.camsig.hdf5")
    cam_data = _c.CamData(signal, params)
    with _bpv.figure() as (_, axes):
        cam_data.plot_cam(axes[0])
    out = None
    if cam_data.anomalies.size == 0 and ask_continue():
        cg = _c.OpenSCADCamGenerator(cam_data, config.cam_folder)
        if (inst_folder := (cg.folder / "inst")).exists() and any(
            inst_folder.iterdir()
        ):
            raise WorkflowError("app.cam.has-inst-cannot-overwrite")
        on_stl_started()
        out = cg.run(force=force)
        on_stl_finished()
    return out and out.relative_to(config.simulator_home)


def register_cam_instance(cam_folder: _pl.Path, inst: _c.CamInstance) -> _pl.Path:
    """Registers a physical cam instance for usage in measurements.

    :param cam_folder: the folder of the cam model relative to the cam folder,
        e. g. ``AAC276_4/d605r3000a100rf0/SC.1.0``.
    :param inst: cam instance metadata.
    :return: the path of the metadata file.
    """
    config = read_config()
    folder = config.cam_folder / cam_folder / "inst"
    folder.mkdir(exist_ok=True)
    seq = max([int(f.name) for f in folder.iterdir() if f.is_dir()] or [0]) + 1
    out_path = folder / str(seq) / "inst.json"
    out_path.parent.mkdir(exist_ok=False)
    inst.seq = seq
    with out_path.open("w") as f:
        print(inst.model_dump_json(indent=4), file=f)
    return out_path


def create_setup(
    hw_version: str,
    setup_label: str,
) -> _pl.Path:
    """Creates the folder and a ``simulator_setup.json`` description file
    for measurements for a particular measurement setup.

    .. note::
        Edit ``simulator_setup.json`` manually after running this command.

    :param hw_version: hardware version of the simulator.
    :param setup_label: unique identifier of the setup.
    :return: the path of ``simulator_setup.json``
    :raises WorkflowError: if the setup already exists.
    """
    config = read_config()
    setup_path = config.measurement_folder / hw_version / setup_label
    if setup_path.exists():
        raise WorkflowError("app.setup.already-exists")
    setup_path.mkdir(parents=True, exist_ok=False)
    config_path = setup_path / "simulator_setup.json"
    with config_path.open("w") as f:
        print(
            _m.SimulatorParams(
                label=setup_label,
                lever_class=2,
                lever_r1=1,
                lever_r2=1,
                fulcrum_x=1,
                cam_follower_d=1,
                cam_follower_h=0,
                lever_follower_d=0,
                cams={"!!!SAMPLE!!!": "EDIT_ME/d615r3000a100rf0/SC.1.0/inst/1"},
            ).model_dump_json(indent=4),
            file=f,
        )
    return config.relative(config_path)


def draw_config(setup_path: _pl.Path, *, r_cam: float) -> _plt.Axes:
    """Visualizes the current measurement setup parameters.

    :param setup_path: full path of the measurement setup folder.
    :param r_cam: the baseline radius of the cam.
    """
    config_path = setup_path / "simulator_setup.json"
    with config_path.open() as f:
        config = _m.SimulatorParams(**json.load(f))
        return config.draw(r_cam=r_cam)


def parse_measurement_name(in_path: _pl.Path) -> _m.MeasParams:
    """Extracts measurement parameters from the conventional file name.

    :raises WorkflowError: if the filename cannot be parsed.
    """
    stem = in_path.stem
    if m := re.match(
        r"^(?P<protocol>\w+)__(?P<cam>\w+)__(?P<u>\d+(?:_\d+)?)V__.+$", stem
    ):
        return _m.MeasParams(
            protocol=m["protocol"], u=float(m["u"].replace("_", ".")), cam_inst=m["cam"]
        )
    else:
        raise WorkflowError("app.meas.unparsable-filename")


def convert_measurement_log(
    *,
    reader: _bp.SignalReader,
    in_path: _pl.Path,
    meas_setup_folder: _pl.Path,
    confirm: _t.Callable[[_m.MeasParams], bool],
    force: bool = False,
) -> tuple[_pl.Path, _m.MeasWithMeta]:
    """Converts a raw measurement log to the uniformised format
    containig measurement metadata as well.

    :param reader: a :class:`bpwave.SignalReader` instance responsible for
        converting the log to a :class:`bpwave.Signal` object.
    :param in_path: full path of the log.
    :param meas_setup_folder: the target setup folder, see :func:`create_setup`.
    :param confirm: confirm callback for measurement parameters.
    :param force: overwrite existing converted file.
    :return: output path and the signal with metadata.
    """
    config = read_config()
    import_res = import_measurement_log(
        reader=reader,
        in_path=in_path,
        confirm=confirm,
    )
    meas_env = collect_measurement_meta(in_path, import_res.meas_params)
    all_data = _m.MeasWithMeta(
        measured=import_res.signal,
        meas_params=import_res.meas_params,
        **meas_env.__dict__,
    )
    out_path = (
        config.measurement_folder
        / meas_setup_folder
        / in_path.with_suffix(".m.hdf5").name
    )
    if out_path.exists() and not force:
        raise WorkflowError("app.signal.already-exists")
    with h5py.File(out_path, "w") as f:
        all_data.to_hdf(f)
    return config.relative(out_path), all_data


@_dc.dataclass(kw_only=True)
class ImportLogResult:
    """Result of :func:`import_measurement_log`."""

    signal: _bp.Signal
    """The converted signal."""

    other_data: _t.Any
    """Other data returned by the signal reader."""

    meas_params: _m.MeasParams
    """Measurement parameters extracted from the filename."""


def import_measurement_log(
    *,
    reader: _bp.SignalReader,
    in_path: _pl.Path,
    confirm: _t.Callable[[_m.MeasParams], bool],
) -> ImportLogResult:
    """Converts a sensor output file to a :class:`bpwave.Signal` and
    extracts measurement parameters from the filename.

    :param reader: a specialized reader for the given file format.
    :param in_path: full path of the sensor output file.
    :param confirm: callback to ask for confirmation of measurement parameters.
    """
    config = read_config()
    meas_params = parse_measurement_name(in_path)
    if not confirm(meas_params):
        raise NotConfirmed("import_measurement_log")
    signal, other = reader(in_path, rel_to=config.simulator_home)
    signal.meta["simulator_meas_params"] = meas_params.model_dump_json()

    return ImportLogResult(
        signal=signal,
        other_data=meas_params,
        meas_params=meas_params,
    )


def collect_measurement_meta(
    meas_path: _pl.Path,
    meas_params: _m.MeasParams,
) -> _m.MeasEnvironment:
    """Collects all measurement metadata from the application folder so that the
    measurement file be self-contained ad the measurement be interpretable
    without the full environment.

    :param meas_path: the path of the measurement file relative to
        ``<app home>/meas``.
    :param meas_params: measurement parameters
    :return: an object containing all data
    """
    app_config = read_config()
    with open(
        app_config.measurement_folder / meas_path.parent / "simulator_setup.json"
    ) as pj:
        sim_params = _m.SimulatorParams(**json.load(pj))
    cam_inst_folder = app_config.cam_folder / sim_params.cams[meas_params.cam_inst]
    cam_folder = cam_inst_folder.parent.parent
    with h5py.File(next(cam_folder.glob("*.cam.hdf5"))) as f:
        cam_signal_nom = _bp.Signal.from_hdf(f["orig_signal"])
    with open(next(cam_folder.glob("*.cam.json"))) as pj:
        cam_params = _c.CamParams(**json.load(pj))
    with open(cam_inst_folder / "inst.json") as pj:
        cam_inst = _c.CamInstance(**json.load(pj))

    return _m.MeasEnvironment(
        nominal=cam_signal_nom,
        cam_params=cam_params,
        cam_inst=cam_inst,
        simulator=sim_params,
        cam_params_source=app_config.relative(cam_folder),
        cam_inst_source=app_config.relative(cam_inst_folder),
    )


DFT_VALIDATION_NOTEBOOK: _t.Final[_pl.Path] = (
    _pl.Path(__file__).parent / "meas_nb" / "meas_vs_cam.ipynb"
)
"""Path of the default validation notebook."""


def run_validation_notebook(
    measurement_paths: list[_pl.Path],
    notebook: _pl.Path = DFT_VALIDATION_NOTEBOOK,
    *,
    absolute: bool = False,
    on_start: _t.Callable[[_pl.Path], None] | None = None,
    on_ready: _t.Callable[[_pl.Path], None] | None = None,
    on_skipped: _t.Callable[[_pl.Path], None] | None = None,
    force: bool = False,
) -> set[_pl.Path]:
    """Executes parameterized copies of a validation notebook on
    measurement files (``.m.hdf5``).

    The validation notebook must have the following parameters:

    * ``par_meas_file``
    * ``par_results_folder``

    The outputs will be at paths ``<measurement file's folder>/<notebook name>/<meas>.*``,
    where ``<meas>`` is the stem of the measurement file name.
    The outputs consist of the notebook and any other files generated when executing it.

    :param measurement_paths: list of paths (can be glob patterns).
    :param notebook: path of the validation notebook,
        by default the one provided in this package (:data:`DFT_VALIDATION_NOTEBOOK`).
    :param absolute: if true, ``measurement_paths`` are assumed to be absolute,
        otherwise they are relative to ``<app home>/meas`` (default).
    :param on_start: callback with path.
    :param on_ready: callback with path.
    :param on_skipped: callback with path.
    :param force: overwrite existing output notebooks.
    :return: set of output paths.
    """
    config = read_config()
    paths = (
        measurement_paths
        if absolute
        else [config.measurement_folder / path for path in measurement_paths]
    )
    paths = [_pl.Path(p) for pattern in paths for p in glob.glob(str(pattern))]
    results = set()
    for path in paths:
        if on_start:
            on_start(path)
        result_folder = path.parent / notebook.stem
        result_folder.mkdir(exist_ok=True)
        output_path = result_folder / f"{path.stem}.ipynb"
        if not output_path.exists() or force:
            papermill.execute_notebook(
                notebook,
                output_path,
                parameters=dict(
                    par_meas_file=str(path),
                    par_results_folder=str(result_folder),
                ),
            )
            results.add(output_path)
            if on_ready:
                on_ready(path)
        else:
            if on_skipped:
                on_skipped(path)
    return results


DFT_SUMMARY_NOTEBOOK: _t.Final[_pl.Path] = (
    _pl.Path(__file__).parent / "meas_nb" / "meas_vs_cam_fcr_waveform_summary.ipynb"
)
"""Path of the default validation summary notebook."""


def run_validation_summary_notebook(
    result_folders: list[_pl.Path],
    notebook: _pl.Path = DFT_SUMMARY_NOTEBOOK,
    *,
    tag: str = "*",
    absolute: bool = False,
    on_start: _t.Callable[[], None] | None = None,
    on_ready: _t.Callable[[], None] | None = None,
    on_skipped: _t.Callable[[], None] | None = None,
    force: bool = False,
) -> _pl.Path | None:
    """Executes a parameterized copy of the validation summary notebook
    on one or more folders containing validation result files (``.r.hdf5``).

    The content of these files is not specified by this package, it depends
    on the validation notebook you have executed.

    The summary notebook must have the following parameters:

    * ``par_result_folders``: comma-separated string of validation result folder paths
    * ``par_tag``: measurement tag

    :param result_folders: folders containing validation results
    :param notebook: full path of a validation summary notebook,
        by default :data:`DFT_SUMMARY_NOTEBOOK`.
    :param tag: measurement tag, the last portion of the conventional
        measurement file name, e. g. in case of
        ``1H__AAC27_rf50_Phr1__12V__240207_1h.1.m.hdf5`` tag is ``240207_1h``.
        If tag is ``*``, no filtering is performed on tags.
    :param absolute: if true, ``measurement_paths`` are assumed to be absolute,
        otherwise they are relative to ``<app home>/meas`` (default).
    :param on_start: callback with path.
    :param on_ready: callback with path.
    :param on_skipped: callback with path.
    :param force: overwrite existing output notebook.
    :return: set of output paths.
    """
    config = read_config()
    paths = (
        result_folders
        if absolute
        else [config.measurement_folder / path for path in result_folders]
    )
    paths = [_pl.Path(p) for pattern in paths for p in glob.glob(str(pattern))]
    if on_start:
        on_start()
    output_path = paths[0] / notebook.name
    if not output_path.exists() or force:
        papermill.execute_notebook(
            notebook,
            output_path,
            parameters=dict(
                par_result_folders=",".join(map(str, paths)),
                par_tag=tag or "*",
            ),
        )
        if on_ready:
            on_ready()
        return output_path
    else:
        if on_skipped:
            on_skipped()
        return None
