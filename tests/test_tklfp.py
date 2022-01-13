import numpy as np
import matplotlib.pyplot as plt
import pytest

from tklfp import TKLFP


def test_horizontal_profile():
    for is_exc in [True, False]:  # both exc and inh
        for z_mm in [-0.1, 0, 0.4]:  # different depths for neuron
            # electrode coords are at 0, 1, and 2 mm horizontal distance
            tklfp = TKLFP([0], [0], [z_mm], [is_exc], [
                          [0, 0, 0], [1, 0, 0], [0, 2, 0]])
            lfp = tklfp.compute([0], [0], [15])
            # smaller the further away it is horizontally
            assert np.abs(lfp[0, 0]) > np.abs(lfp[0, 1]) > np.abs(lfp[0, 2])
            # electrode 1 mm away from spike at origin should be same as spike 1 mm away
            # from electrode at origin
            assert lfp[0, 1] == TKLFP([0], [1], [z_mm], [is_exc]).compute(
                i_spikes=[0], t_spikes_ms=[0], t_eval_ms=[15]
            )


def test_time_profile():
    for is_exc in [True, False]:  # both exc and inh
        # electrodes at .1, 0, and -.4 mm test depth profile at -.1, 0, and .4 mm
        tklfp = TKLFP([0], [0], [0], [is_exc], [
                      [0, 0, 0.1], [0, 0, 0], [0, 0, -0.4]])
        # eval times should get amp before, at, after, and long after peak
        lfp = tklfp.compute([0], t_spikes_ms=[0], t_eval_ms=[5, 10.4, 15, 20])
        for col in range(3):
            assert (
                np.abs(lfp[0, col])  # before
                < np.abs(lfp[1, col])  # peak
                > np.abs(lfp[2, col])  # after
                > np.abs(lfp[3, col])  # long after
            )

d = 0.4

def _plot_test(t1, t2, y1, z1):
    """Useful for visualizing test cases in following window test"""
    t = np.linspace(0, 25)
    lfp1 = TKLFP([0], [y1], [z1], t1).compute([0], [0], t)
    lfp2 = TKLFP([0], [0], [0], t2).compute([0], [0], t)
    fig, ax = plt.subplots()
    ax.plot(t, lfp1)
    ax.plot(t, lfp2)


@pytest.mark.parametrize(
    "type1, type2, y1, z1, win1_gt_win2",
    [
        # equal, so not win1 not greater than win2
        ("e", "e", 0, 0, False),
        # neuron1 has lower amp but later peak
        ("e", "e", d, 0, True),
        # window smaller with only vertical distance since same time but smaller
        ("e", "e", 0, d, False),
        ("i", "i", 0, 0, False),  # equal
        ("i", "i", d, 0, True),
        ("i", "i", 0, d, False),
        ("i", "e", 0, 0, False),
        # neuron1 has later peak but lower amplitude and spread
        ("i", "e", d, 0, False),
        # neuron1 has bigger amplitude but narrower spread 
        ("i", "e", 0, d, False),
        ("e", "i", 0, 0, True),  # because of wider temporal spread
        ("e", "i", d, 0, True),  # because of wider temporal spread
        ("e", "i", 0, d, True),  # because of wider temporal spread
    ],
)
def test_min_window_type_and_distance(type1, type2, y1, z1, win1_gt_win2):
    is_exc1 = type1 == "e"
    is_exc2 = type2 == "e"
    win1 = TKLFP([0], [y1], [z1], is_exc1).compute_min_window_ms(1e-3)
    win2 = TKLFP([0], [0], [0], is_exc2).compute_min_window_ms(1e-3)
    assert (win1 > win2) == win1_gt_win2

@pytest.mark.parametrize("is_exc", [True, False], ids=["exc", "inh"])
def test_min_window_threshold(is_exc):
    thresholds = [10**p for p in range(-10, 3)]
    tklfp = TKLFP([0], [0], [0], is_exc)
    windows = np.asarray([tklfp.compute_min_window_ms(th) for th in thresholds])
    # window widths should monotonically decrease as the threshold increases
    assert all(np.diff(windows) <= 0)
    # giant thresholds (10, 100, over any uLFP peaks) should produce 0
    assert all(windows[-2:] == 0)

