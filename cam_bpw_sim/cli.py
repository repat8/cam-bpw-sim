import pathlib as _pl
from typing import Annotated, Optional

import click
import matplotlib.pyplot as _plt
import typer as _ty
from typer import Argument, Option

from . import __version__, app
from . import cam as _c
from .sensors import optoforce

cli_app = _ty.Typer()


def _version_callback(value: bool) -> None:
    if value:
        _ty.echo(f"{app.APP_NAME} v{__version__}")
        raise _ty.Exit()


@cli_app.callback()
def main(
    version: Optional[bool] = Option(
        None,
        "--version",
        "-v",
        help="Show the application's version and exit.",
        callback=_version_callback,
        is_eager=True,
    )
) -> None:
    return


@cli_app.command()
def init_config() -> None:
    """Create application config file at the user's config folder."""
    app.init_app_config(on_exists=_ty.echo, on_create=_ty.echo)


@cli_app.command()
def init_home() -> None:
    """Init application home folder structure at the location specified
    by the config.
    """
    app.init_simulator_home(
        on_home=lambda p: _ty.echo(f"Home: {p}"),
        on_create=lambda p: _ty.echo(f"Created {p}"),
    )


@cli_app.command()
def physionet(
    record: str,
    start: Annotated[int, Option()] = 0,
    stop: Optional[int] = None,
    force: bool = False,
) -> None:
    """Download BP signal from the Autonomic Aging dataset of Physionet."""
    path, signal = app.download_physionet_autonomic_aging(
        record=record,
        start=start,
        stop=stop,
        force=force,
    )
    signal.plot(title=path.name)
    _ty.echo(str(path))
    _plt.show(block=True)


@cli_app.command()
def list_raw() -> None:
    """List available raw signals."""
    for name in app.list_raw_signals():
        _ty.echo(name)


@cli_app.command()
def list_prep() -> None:
    """List available preprocessed signals."""
    for name in app.list_prep_signals():
        _ty.echo(name)


@cli_app.command()
def view_signal(rel_path: _pl.Path) -> None:
    """Visualize a signal."""
    app.view_signal(rel_path)
    _plt.show(block=True)


@cli_app.command()
def chpoints(signal_name: str) -> None:
    """Create a preprocessed signal by detecting characteristic points."""
    out, _ = app.prep_detect_chpoints(signal_name)
    _ty.echo(out)
    _plt.show(block=True)


@cli_app.command()
def cam_signal(
    base_name: str,
    cam_signal_name: str,
    from_onset: Annotated[int, Option("-0", "--from-onset")],
    n_ccycles: Annotated[int, Option("-n", "--n-ccycles")],
) -> None:
    """Create a preprocessed signal suitable for cam generation."""
    out, _ = app.prep_for_cam(
        base_name, cam_signal_name, from_onset=from_onset, n_ccycles=n_ccycles
    )
    _ty.echo(out)
    _plt.show(block=True)


@cli_app.command()
def cam_stl(
    name: str,
    amplitude: Annotated[float, Option("-A", "--amplitude")],
    r: Annotated[float, Option("-r", "--r-cam")],
    d_follower: Annotated[float, Option("-F", "--d-follower")],
    d_shaft: Annotated[float, Option("-d", "--d-shaft")],
    rotation: _c.Rotation = _c.Rotation.ccw,
    invert: bool = False,
    force: bool = False,
) -> None:
    """Generate an STL file from a preprocessed cam signal."""
    path = app.generate_cam(
        _c.CamParams(
            name=name,
            amplitude=amplitude,
            r=r,
            r_rim=r - 1,
            r_follower=d_follower / 2,
            d_shaft=d_shaft,
            rotation=rotation,
            invert=invert,
        ),
        force=force,
        ask_continue=lambda: _ty.confirm("Continue?"),
        on_stl_started=lambda: _ty.echo("Generating STL started..."),
        on_stl_finished=lambda: _ty.echo("STL is ready."),
    )
    _ty.echo(path)
    _plt.show(block=True)


@cli_app.command()
def cam_inst(cam_folder: _pl.Path) -> None:
    """Register a physical cam instance."""
    config = app.read_config()
    inst = _c.CamInstance(
        **{
            name: (
                _ty.prompt(
                    name,
                    type=click.Choice(getattr(config.cam_instance, name)),
                    show_choices=True,
                )
                if hasattr(config.cam_instance, name)
                else _ty.prompt(name, default=field.default, show_default=True)
            )
            for name, field in _c.CamInstance.model_fields.items()
            if name != "seq"
        }
    )
    _ty.confirm("OK?", abort=True)
    out_path = app.register_cam_instance(cam_folder, inst)
    _ty.echo(out_path)


@cli_app.command()
def create_setup(
    hw_version: Annotated[str, Argument(help="Hardware version")],
    setup_label: Annotated[str, Argument(help="Unique identifier of setup")],
) -> None:
    """Register a new measurement setup."""
    path = app.create_setup(
        hw_version=hw_version,
        setup_label=setup_label,
    )
    _ty.echo(f"Please edit {path}")


@cli_app.command()
def draw_setup(
    meas_setup_folder: _pl.Path,
    r_cam: Annotated[float, Option(help="Cam radius", show_default=True)] = 30.0,
) -> None:
    """Display a schematic drawing of the simulator given a measurement setup."""
    app.draw_config(meas_setup_folder, r_cam=r_cam)
    _plt.show()


@cli_app.command()
def import_optoforce(
    in_path: _pl.Path,
    meas_setup_folder: _pl.Path,
    fs: Annotated[float, Option()],
    channel: Optional[int] = None,
    force: bool = False,
) -> None:
    """Import a measurement log from an OptoForce 3D force sensor to a
    measurement setup."""

    def confirm(m):
        _ty.echo(m.model_dump_json(indent=4))
        return _ty.confirm("Are these parameters correct?")

    out_path, data = app.convert_measurement_log(
        reader=optoforce.OptoForceCsvReader(fs_override=fs, channel=channel),
        in_path=in_path,
        meas_setup_folder=meas_setup_folder,
        confirm=confirm,
        force=force,
    )
    _ty.echo(out_path)


@cli_app.command()
def meas_val(
    meas_files: list[_pl.Path],
    absolute: bool = False,
    force: bool = False,
    notebook: Optional[_pl.Path] = app.DFT_VALIDATION_NOTEBOOK,
) -> None:
    """Run measurement validation with imported measurement files."""
    paths = app.run_validation_notebook(
        meas_files,
        notebook=notebook or app.DFT_VALIDATION_NOTEBOOK,
        absolute=absolute,
        force=force,
        on_start=lambda p: _ty.echo(f"Started: {p}"),
        on_skipped=lambda p: _ty.secho(f"Skipped: {p}", fg=_ty.colors.YELLOW),
        on_ready=lambda p: _ty.secho(f"Ready: {p}", fg=_ty.colors.GREEN),
    )
    if not paths:
        _ty.secho("No outputs.", fg=_ty.colors.YELLOW)
    else:
        _ty.secho("Outputs:", bold=True)
        for p in paths:
            _ty.echo(str(p))
        _ty.secho("Output folder:", bold=True)
        _ty.echo(str(list(paths)[0].parent))


@cli_app.command()
def meas_val_summary(
    result_folders: list[_pl.Path],
    notebook: Annotated[_pl.Path, _ty.Option()] = app.DFT_SUMMARY_NOTEBOOK,
    tag: Annotated[str, _ty.Option()] = "*",
    absolute: bool = False,
    force: bool = False,
) -> None:
    """Calculate aggregated statistics on measurement validations."""
    path = app.run_validation_summary_notebook(
        result_folders,
        notebook=notebook,
        tag=tag,
        absolute=absolute,
        force=force,
        on_start=lambda: _ty.echo("Started"),
        on_skipped=lambda: _ty.secho("Skipped", fg=_ty.colors.YELLOW),
        on_ready=lambda: _ty.secho("Ready", fg=_ty.colors.GREEN),
    )
    if not path:
        _ty.secho("No outputs.", fg=_ty.colors.YELLOW)
    else:
        _ty.secho("Output:", bold=True)
        _ty.echo(str(path))


def run_cli():
    cli_app(prog_name=app.APP_NAME)
