import numpy as np
import matplotlib.pyplot as plt
import pytest
import wslfp as WSLFP

def test_horizontal_profile():
    #if the electrode is closer to the neuron, signal should be stronger
    #x_mm = [0]
    #y_mm = [0]
    #z_mm = [0]

    #electrode = [[0, 0, 0], [1, 0, 0], [0, 2, 0]]

    for z_mm in [-0.1, 0, 0.2]: # test different depth for neurons
        wslfp = WSLFP([0], [0], [z_mm], [[0, 0, 0], [1, 0, 0], [0, 2, 0]])
        lfp = WSLFP.compute([0], [5], [0], [5], [15])
        # should be smaller the further away it is horizontally
        assert np.abs(lfp[0, 0]) > np.abs(lfp[0, 1]) > np.abs(lfp[0,2])
        assert lfp[0,1] == WSLFP([0], [1], [z_mm]).compute(ampa = [0], t_ampa_ms = [10], gaba = [0], t_gaba_ms = [10], t_eval_ms = [15])

""" def test_depth_profile():
    # test with electrode contacts at 4 canonical depths
    # lfp = WSLFP([0], [0], [0], elec_coords_mm=[[0, 0, -0.4], [0, 0, 0], [0, 0, 0.4], [0, 0, 0.8]],
    # ).compute(
    #     [0], [5], [0], [5], [15]]
    # )
    lfp_amp = wslfp.amplitude(neuron_coords, elec_coords_mm=[[0, 0, -0.4], [0, 0, 0.4], [0, 0, 0.8]])
    positive = [True, False, False]
    assert np.all((lfp_amp > 0) == positive)
    increasing = [False, True]
    assert np.all((np.diff(lfp_amp) > 0) == increasing) """

def test_time_profile(increasing):
    # signal should increase over time because it is weighted sum
    for t_eval_ms in [12, 13, 14, 15, 16]:
        lfp = WSLFP([0], [0], [0], elec_coords_mm=[[0, 0, -0.4], [0, 0, 0], [0, 0, 0.4], [0, 0, 0.8]])
        lfp = WSLFP.compute([0], [5], [0], [5], [t_eval_ms])
        #should be greater as time increases
        assert np.all((np.diff(lfp) > 0) == increasing)

@pytest.mark.parametrize (
    "t_ampa, t_gaba, t_eval, success",
    ([10], [4], [10], True),
    ([5], [5], [10], False),
    ([10], [4.1], [10], False),
    ([9.9], [4], [10], False),
    ([9.9, 10.1], [4], [10], True),
    ([11], [5], [11], True),
    ([2], [8], [6], False),
)

def test_check_timepoints(t_ampa, t_gaba, t_eval, success):
    #t_ampa_ms needs to be greater than 6 ms before the last eval point
    if not success:
        with pytest.raises(Exception):
            lfp = WSLFP._check_timepoints(t_ampa, t_gaba, t_eval);
    else:
        lfp = WSLFP._check_timepoints(t_ampa, t_gaba, t_eval);

    # need to check multiple timepoints: mulitple AMPA, GABA, and eval times
    # need to make sure exact time works

    # dont need to check every single time points

    