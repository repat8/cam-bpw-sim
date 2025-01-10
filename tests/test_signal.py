import bpwave
import matplotlib.pyplot as plt
import numpy as np
import pytest

from cam_bpw_sim.signal import LocMinDetector, norm01

# def test_spline_corr(plot: bool = False):
#     bp = utils.filtered_bp_signal()
#     bp.y = rbp.pre.standardize(bp.y)
#     onset_res = rbp.segment.Shin2009PointDetector(find_max=False)(bp)
#     onset_res.insert_into(bp, set_onsets=True)
#     corr = CubicSplineCorr()
#     corr_res = corr(bp, show=plot)
#     if plot:
#         corr_res.plot(bp, title="test_spline_corr")
#         plt.show()
#     assert corr_res.y_corr.shape == bp.y.shape
#
#
# @pytest.mark.human
# def test_spline_corr_plot():
#     test_spline_corr(True)


def test_loc_min_detector(plot: bool = False):
    t = np.linspace(-0.5 * np.pi, 3.5 * np.pi, 3 * 100)
    bp = bpwave.Signal(y=np.sin(t), t=t)
    det = LocMinDetector(refrac=0.5)
    onsets = det(bp, show=plot)
    if plot:
        onsets.plot(bp, title="test_loc_min_detector")
        plt.show()
    assert all([i.onset >= 0 for i in onsets.chpoints.indices])
    assert [i.onset for i in onsets.chpoints.indices], [0, len(bp.y) // 2, len(bp.y)]
    assert len(onsets.chpoints.indices) == 3


@pytest.mark.human
def test_loc_min_detector_plot():
    test_loc_min_detector(True)


def test_norm01__positive():
    n = norm01(np.array([1, 5, 3, 1]))
    assert n.tolist() == [0.0, 1.0, 0.5, 0.0]


def test_norm01__negative():
    n = norm01(-np.array([1, 5, 3, 1]))
    assert n.tolist() == [1.0, 0.0, 0.5, 1.0]
