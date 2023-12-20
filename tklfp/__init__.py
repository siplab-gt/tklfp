"""Lightweight implementation of Telenczuk 2020 kernel LFP approximation"""
import pickle
from typing import Union

import numpy as np
import scipy  # noqa: F401
from importlib_resources import files
from numpy.typing import ArrayLike


def _load_uLFP_A0_profile(fname):
    # with open_binary(__package__, fname) as f:
    with files(__package__).joinpath(fname).open("rb") as f:
        A0_profile = pickle.load(f)

    return A0_profile


######### PARAMETERS #############
_sig_i = 2.1
params2020 = {
    "va_m_s": 0.2,  # axonal velocity (m/sec)
    "lambda_mm": 0.2,  # space constant (mm)
    "sig_i_ms": _sig_i,  # std-dev of ihibition (in ms)
    "sig_e_ms": 1.5 * _sig_i,  # std-dev for excitation
    "d_ms": 10.4,  # constant delay
    "exc_A0_by_depth": _load_uLFP_A0_profile("exc_A0_by_depth.pkl"),
    "inh_A0_by_depth": _load_uLFP_A0_profile("inh_A0_by_depth.pkl"),
}


class TKLFP:
    """Implements kernel LFP approximation"""

    def __init__(
        self,
        xs_mm: ArrayLike,
        ys_mm: ArrayLike,
        zs_mm: ArrayLike,
        is_excitatory: Union[ArrayLike, bool],
        elec_coords_mm: ArrayLike = [[0, 0, 0]],
        orientation: ArrayLike = [0, 0, 1],
        params: dict = params2020,
    ) -> None:
        """Constructor: caches per-spike contributions to LFP for given neurons.

        Parameters
        ----------
        xs_mm : ArrayLike
            Sequence of length N_n, contains X coordinates of N_n neurons in mm
        ys_mm : ArrayLike
            Sequence of length N_n, contains Y coordinates of N_n neurons in mm
        zs_mm : ArrayLike
            Sequence of length N_n, contains Z coordinates of N_n neurons in mm
        is_excitatory : ArrayLike | bool
            Sequence of length N_n, contains cell type of N_n neurons where
            False (0) represents inhibitory and True (1) represents excitatory.
            Can also be simply True or False if all neurons have same type.
        elec_coords_mm : ArrayLike, optional
            Shape (N_e, 3), where N_e is the number of recording sites and the
            three columns represent X, Y, and Z coordinates.
            By default [[0, 0, 0]].
        orientation : ArrayLike, optional
            Represents which direction is "up"—towards the surface of the
            cortex—for the neurons. It can be a 1x3 vector if orientation is
            uniform, else an array of shape `(N_n, 3)`. [0, 0, 1] by default.
        params : dict, optional
            Dict containing parameters. See the default params2020 object for
            required elements
        """

        assert len(xs_mm) == len(ys_mm) == len(zs_mm)
        n_nrns = len(xs_mm)
        if is_excitatory is not np.ndarray:
            # reshape to ensure it's a 1D array
            is_excitatory = np.array(is_excitatory).reshape((-1,))
        if len(is_excitatory) == 1:
            is_excitatory = is_excitatory.repeat(n_nrns)
        is_excitatory = is_excitatory.astype(bool)
        assert len(is_excitatory) == n_nrns
        if type(elec_coords_mm) is not np.ndarray:
            elec_coords_mm = np.array(elec_coords_mm)
        assert elec_coords_mm.shape[1] == 3
        ornt_shape = np.shape(orientation)
        assert len(ornt_shape) in [1, 2] and ornt_shape[-1] == 3

        # normalize orientation vectors
        orientation = orientation / np.linalg.norm(orientation, axis=-1, keepdims=True)

        # calc ampltiude and delay for each neuron for each contact
        # height h is "depth" in paper, in axis defined by apical dendrite
        # r is radius: distance perpendicular to h
        # d is distance from neuron to electrode
        # + --- r --- * <- electrode
        # |         /
        # |       /
        # h     dist
        # |   /
        # | /
        # Δ  <- neuron (pointing up if pyramidal cell)
        n_elec = elec_coords_mm.shape[0]
        dist = np.tile(np.expand_dims(elec_coords_mm, axis=1), (1, n_nrns, 1)).astype(
            "float64"
        )  # n_elec X n_nrns X 3
        dist[:, :, 0] -= xs_mm
        dist[:, :, 1] -= ys_mm
        dist[:, :, 2] -= zs_mm

        # theta = arccos(o*d/(||o||*||d||))
        norm_dist = np.linalg.norm(dist, axis=2)  # shape (n_elec, n_nrns)
        # since 0 dist leads to division by 0 and numerator of 0 is "invalid"
        old_settings = np.seterr(divide="ignore", invalid="ignore")
        theta = np.nan_to_num(
            np.arccos(
                np.sum(
                    orientation * dist, axis=2
                )  # multiply elementwise then sum across x,y,z to get dot product
                / (1 * norm_dist)  # norm of all orientation vectors should be 1
            )
        )  # shape (n_elec, n_nrns)
        np.seterr(**old_settings)

        h = norm_dist * np.cos(theta)  # shape (n_elec, n_nrns)
        r = norm_dist * np.sin(theta)  # shape (n_elec, n_nrns)

        # d = np.sqrt(np.sum(d**2, axis=1))  # n_elec X n_nrns

        # originally used lateral distance r here, but the paper
        # actually just says "distance". so using ||d|| instead
        # dist in mm, va in m/s, so dist/va will be in ms
        self._delay = params["d_ms"] + norm_dist / params["va_m_s"]
        # self._delay is n_elec X n_nrns

        # amplitude and width of kernel depend on cell type
        A0 = np.zeros(((n_elec, n_nrns)))
        # 2 sigma squared, used in Gaussian kernel
        self._ss = np.ones(n_nrns)
        A0[:, is_excitatory] = params["exc_A0_by_depth"](h[:, is_excitatory])
        A0[:, ~is_excitatory] = params["inh_A0_by_depth"](h[:, ~is_excitatory])
        self._ss[is_excitatory] = 2 * params["sig_e_ms"] ** 2
        self._ss[~is_excitatory] = 2 * params["sig_i_ms"] ** 2

        # originally used lateral distance r here, but the paper
        # actually just says "distance". so using ||d|| instead
        self._amp = A0 * np.exp(-norm_dist / params["lambda_mm"])
        # self._amp is also n_elec X n_nrns
        self.params = params

    def compute(
        self,
        i_spikes: ArrayLike,
        t_spikes_ms: ArrayLike,
        t_eval_ms: ArrayLike,
    ) -> np.ndarray:
        """Computes the tklfp for given spikes at desired timepoints.

        Parameters
        ----------
        i_spikes : ArrayLike[int]
            Neuron indices of spikes. Must be between 0 and N_n,
            corresponding to the parameters given on initialization.
        t_spikes_ms : ArrayLike[float]
            Times (in ms) of spikes. Must have same length as i_spikes.
        t_eval_ms : ArrayLike[float]
            Times (in ms) at which to evaluate LFP.

        Returns
        -------
        tklfp : [np.ndarray]
            An N_eval by N_elec array containing the computed tklfp
            with one row for each timepoint and one column for each
            recording site.
        """
        for arg in [i_spikes, t_spikes_ms, t_eval_ms]:
            assert isinstance(arg, (list, np.ndarray, tuple))
        # get values needed for neurons that spiked
        amp = self._amp[:, i_spikes]  # will be n_elec X n_spikes
        delay = self._delay[:, i_spikes]  # ⬆ same dim as above ⬆
        n_elec = amp.shape[0]
        ss = self._ss[i_spikes]  # n_spikes

        if not isinstance(t_eval_ms, np.ndarray):
            t_eval_ms = np.array(t_eval_ms)
        n_eval = len(t_eval_ms)

        # will be n_eval X n_elec X n_spikes. can be broadcast with amp, delay, ss by aligning
        # last dims:       n_elec X n_spikes
        t = (
            np.tile(t_eval_ms.reshape(n_eval, 1, 1), (1, n_elec, len(t_spikes_ms)))
            - t_spikes_ms
            - delay
        )
        # multiply amplitude by temporal kernel:
        contribs = amp * np.exp(-(t**2) / ss)
        # sum over spikes and return. should be n_eval X n_elec
        lfp = np.sum(contribs, axis=2)
        assert lfp.shape == (len(t_eval_ms), n_elec)
        return lfp

    def compute_min_window_ms(self, uLFP_threshold_uV: float):
        """Compute the window required to capture all uLFPs above threshold.

        This is designed to facilitate computing the TKLFP from a buffer
        of fixed width, rather than the entire simulation history. It is
        computed from the single neuron whose uLFP decays to the threshold
        latest after the original spike.

        Parameters
        ----------
        uLFP_threshold_uV : float
            Threshold (in microvolts) above which no single uLFP can be
            ignored. i.e., the window must be wide enough that the uLFP
            from a past spike is captured until it decays to this value.

        Returns
        -------
        float
            The minimum window width, in ms, required to capture all
            uLFPs above the amplitude threshold. If no uLFPs ever
            exceed the threshold, 0 is returned, meaning that no
            window whatsoever is required to capture all supra-
            threshold uLFPs, since there are none.
        """
        # Δ = t_eval - t_peak = t_eval - t_spike - delay
        # ss = 2σ^2
        # uLFP = amp * exp(-Δ^2 / ss)
        # set uLFP equal to threshold and solve:
        # Δ = sqrt(-ss ln(θ/amp))
        # window = t_eval - t_spike = Δ + delay
        # computes window for each neuron: return max
        subthresh = uLFP_threshold_uV > np.abs(self._amp)
        if np.all(subthresh):
            return 0
        with np.errstate(divide="ignore"):
            delta = np.sqrt(
                # take sqrt only above threshold
                (-self._ss * np.log(uLFP_threshold_uV / np.abs(self._amp)))[~subthresh]
            )
        return np.max(delta + self._delay[~subthresh])
