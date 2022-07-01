import numpy as np
import matplotlib.pyplot as plt
import pytest

from tklfp import TKLFP
import tklfp


def test_horizontal_profile():
    for is_exc in [True, False]:  # both exc and inh
        for z_mm in [-0.1, 0, 0.4]:  # different depths for neuron
            # electrode coords are at 0, 1, and 2 mm horizontal distance
            tklfp = TKLFP([0], [0], [z_mm], [is_exc], [[0, 0, 0], [1, 0, 0], [0, 2, 0]])
            lfp = tklfp.compute([0], [0], [15])
            # smaller the further away it is horizontally
            assert np.abs(lfp[0, 0]) > np.abs(lfp[0, 1]) > np.abs(lfp[0, 2])
            # electrode 1 mm away from spike at origin should be same as spike 1 mm away
            # from electrode at origin
            assert lfp[0, 1] == TKLFP([0], [1], [z_mm], [is_exc]).compute(
                i_spikes=[0], t_spikes_ms=[0], t_eval_ms=[15]
            )


# check if signal is positive at different points
# and ensure sigal decreases/increases as expected
@pytest.mark.parametrize(
    "is_exc,positive,increasing",
    [
        (True, [False, True, True, False], [True, False, False]),
        (False, [False, True, False, True], [True, False, True]),
    ],
    ids=["exc", "inh"],
)
def test_depth_profile(is_exc, positive, increasing):
    # test with electrode contacts at 4 canonical depths
    lfp = TKLFP(
        [0],
        [0],
        [0],
        is_exc,
        elec_coords_mm=[[0, 0, -0.4], [0, 0, 0], [0, 0, 0.4], [0, 0, 0.8]],
    ).compute(
        [0], [0], [tklfp.params2020["d_ms"]]  # measure at peak
    )
    assert np.all((lfp > 0) == positive)
    assert np.all((np.diff(lfp) > 0) == increasing)


def test_time_profile():
    for is_exc in [True, False]:  # both exc and inh
        # electrodes at .1, 0, and -.4 mm test depth profile at -.1, 0, and .4 mm
        tklfp = TKLFP([0], [0], [0], [is_exc], [[0, 0, 0.1], [0, 0, 0], [0, 0, -0.4]])
        # eval times should get amp before, at, after, and long after peak
        lfp = tklfp.compute([0], t_spikes_ms=[0], t_eval_ms=[5, 10.4, 15, 20])
        for col in range(3):
            assert (
                np.abs(lfp[0, col])  # before
                < np.abs(lfp[1, col])  # peak
                > np.abs(lfp[2, col])  # after
                > np.abs(lfp[3, col])  # long after
            )


@pytest.mark.parametrize("seed", [1830, 1847])
@pytest.mark.parametrize("n_elec", [1, 3])
@pytest.mark.parametrize("is_exc", [True, False])
def test_orientation_sensitivity(seed, n_elec, is_exc):
    n_nrns = 4
    rng = np.random.default_rng(seed)
    # all neurons in the same spot
    c = np.zeros((n_nrns,))
    # but random orientations
    orientation = rng.random((n_nrns, 3))
    tklfp = TKLFP(c, c, c, is_exc, rng.random((n_elec, 3)), orientation)
    # here the trick is that each neuron should produce sth different since
    # they are all oriented differently
    t_eval_ms = [5, 15]
    results = np.zeros((len(t_eval_ms), n_elec, n_nrns))  # n_eval X n_elec X n_nrns
    for i_nrn in range(n_nrns):
        results[:, :, i_nrn] = tklfp.compute([i_nrn], [0], t_eval_ms)
    # for each neuron, make sure the lfp output doesn't equal that of
    # any of the neurons after it (make sure none match)
    for i_nrn in range(n_nrns - 1):
        # take one slice of results to broadcast and compare to rest
        assert (
            not (results[:, :, i_nrn : i_nrn + 1] == results[:, :, i_nrn + 1 :])
            .all(axis=(0, 1))
            .any()
        )  # condense along time and electrodes before checking for any matches


def _rand_rot_mat(rng: np.random.Generator = np.random.default_rng()):
    τ = 2 * np.pi
    θxy, θyz = τ * rng.random(2)
    # multiply rotation matrices for xy and yz planes together
    rot_mat = np.array(
        [[np.cos(θxy), -np.sin(θxy), 0], [np.sin(θxy), np.cos(θxy), 0], [0, 0, 1]]
    ) @ np.array(
        [[1, 0, 0], [0, np.cos(θyz), -np.sin(θyz)], [0, np.sin(θyz), np.cos(θyz)]]
    )
    return rot_mat.T  # transpose for right instead of left multiplication


@pytest.mark.parametrize("seed", [421, 385])
@pytest.mark.parametrize("n_nrns", [1, 4])
@pytest.mark.parametrize("n_elec", [1, 3])
@pytest.mark.parametrize("is_exc", [True, False])
def test_rotation_invariance(seed, n_nrns, n_elec, is_exc):
    """If orientations work correctly, we should be able to rotate the whole system"""
    rng = np.random.default_rng(seed)
    c = rng.random((n_nrns, 3))
    orientation = rng.random((n_nrns, 3))
    elec_coords = rng.random((n_elec, 3))
    tklfp = TKLFP(c[:, 0], c[:, 1], c[:, 2], is_exc, elec_coords, orientation)
    # now if we rotate both the neurons, orientations, and the electrode coordinates
    # they should have the same relative positions yield the same results every time
    n_spikes = 10
    i_spikes = rng.integers(0, n_nrns, size=(n_spikes,))
    # spikes from 0 to 10 ms
    t_spikes_ms = 10 * rng.random((n_spikes,))
    t_eval_ms = [10, 20, 30]
    lfp = tklfp.compute(i_spikes, t_spikes_ms, t_eval_ms)
    for i_rot in range(6):  # try 6 rotations
        rot_mat = _rand_rot_mat(rng)
        # rotate with right-multiplication for convenience (since mats are nx3)
        c_rot = c @ rot_mat
        tklfp = TKLFP(
            c_rot[:, 0],
            c_rot[:, 1],
            c_rot[:, 2],
            is_exc,
            elec_coords @ rot_mat,
            orientation @ rot_mat,
        )
        assert np.isclose(lfp, tklfp.compute(i_spikes, t_spikes_ms, t_eval_ms)).all()


d = -0.4  # negative coordinate to test positive electrode depth


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
        # equal, so win1 not greater than win2
        ("e", "e", 0, 0, False),
        # neuron1 has lower amp but later peak
        ("e", "e", d, 0, True),
        # neuron1 has later peak, but even lower amp b/c of hor and ver distance
        ("e", "e", 0, d, False),
        # equal
        ("i", "i", 0, 0, False),
        # window bigger with distance, later peak
        ("i", "i", d, 0, True),
        ("i", "i", 0, d, True),
        # greater amp for I than E
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
