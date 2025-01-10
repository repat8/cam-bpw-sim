import pathlib

import numpy as np
import pytest

from cam_bpw_sim.sensors.optoforce import OptoForceCsvReader, fix_optoforce_timestamps


def test__optoforce_csv():
    path = (pathlib.Path(__file__).parent / "optoforce.csv").as_posix()
    reader = OptoForceCsvReader()
    signal, _ = reader(path)
    with open(path) as csv:
        assert len(signal.y) == len([line for line in csv if line]) - 1
        assert signal.y.max() < 1000
        assert signal.meta["sensor_channel"] == "vector_length"


def test__optoforce_csv__fs_override():
    path = (pathlib.Path(__file__).parent / "optoforce.csv").as_posix()
    reader = OptoForceCsvReader(fs_override=333.0, channel=1)
    signal, _ = reader(path)
    with open(path) as csv:
        assert len(signal.y) == len([line for line in csv if line]) - 1
        assert signal.y.min() > 10_000
        assert signal.meta["sensor_channel"] == "cS1"


def test__fix_optoforce_timestamps():
    t_orig = np.ones(5)
    with pytest.warns(UserWarning, match=r"\bfs\b"):
        assert np.allclose(fix_optoforce_timestamps(t_orig), t_orig)
    t_exp = [1.0, 1.1, 1.2, 1.3, 1.4]
    assert np.allclose(fix_optoforce_timestamps(t_orig, fs=10), t_exp)
    t_orig = [1, 1, 2, 2, 2, 2, 3, 3]
    t_exp = [1, 1.5, 2, 2.25, 2.5, 2.75, 3, 3 + 1 / 3]
    assert np.allclose(fix_optoforce_timestamps(t_orig, fs=3), t_exp)
