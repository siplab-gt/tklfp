import numpy as np

from tklfp import TKLFP


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
