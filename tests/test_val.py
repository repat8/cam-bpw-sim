import bpwave
import bpwave.visu
import numpy as np
import pytest

import cam_bpw_sim as bps


def test_match_sampling_highest(plot: bool = False):
    signals = []
    for i, n in enumerate([60, 120, 40]):
        t = np.linspace(0, np.pi * 2, n)
        y = np.sin(t)
        signals.append(bpwave.Signal(t=t, y=y, label=f"s{i}"))
    matched, i_base = bps.val.match_sampling(signals, select="highest")
    if plot:
        with bpwave.visu.figure(
            nrows=2, title="test_match_sampling_highest", block=True
        ) as (_, axes):
            for s in signals:
                s.plot(axes[0])
            for s in matched:
                s.plot(axes[1])
    else:
        assert i_base == 1
        assert np.allclose(matched[0].t, matched[1].t)
        assert np.allclose(matched[0].t, matched[2].t)
        assert np.allclose(matched[i_base].t, signals[i_base].t)


@pytest.mark.human
def test_match_sampling_highest_plot():
    test_match_sampling_highest(True)


def test_match_sampling_lowest(plot: bool = False):
    signals = []
    for i, n in enumerate([60, 120, 40]):
        t = np.linspace(0, np.pi * 2, n)
        y = np.sin(t)
        signals.append(
            bpwave.Signal(
                t=t,
                y=y,
                label=f"s{i}",
                chpoints=bpwave.ChPoints(
                    alg="test",
                    version="0",
                    params={},
                    indices=[bpwave.CpIndices(onset=i) for i in [0, len(y) - 1]],
                ),
                marks={"test_p": [0, len(y) - 1]},
            )
        )
    matched, i_base = bps.val.match_sampling(signals, select="lowest")
    if plot:
        with bpwave.visu.figure(
            nrows=2, title="test_match_sampling_lowest", block=True
        ) as (_, axes):
            for s in signals:
                s.plot(axes[0])
            for s in matched:
                s.plot(axes[1])
    else:
        assert i_base == 2
        assert np.allclose(matched[0].t, matched[1].t)
        assert np.allclose(matched[0].t, matched[2].t)
        assert np.allclose(matched[i_base].t, signals[i_base].t)
        for m in matched:
            assert m.onsets.tolist() == [0, len(m.y) - 1]
            assert m.chpoints.alg == "test"
            assert m.marks["test_p"].tolist() == [0, len(m.y) - 1]


@pytest.mark.human
def test_match_sampling_lowest_plot():
    test_match_sampling_lowest(True)


def test_rmse_perfect():
    s = np.array([1.0, 2.0, 3.0, 4.0])
    err = bps.val.rmse(s, s)
    assert np.allclose(err, 0.0)


def test_rmse_has_error():
    s = np.array([1.0, 2.0, 3.0, 4.0])
    err = bps.val.rmse(s, s + 2)
    assert np.allclose(err, 2.0)
