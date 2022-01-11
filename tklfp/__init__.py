"""Lightweight implementation of Telenczuk 2020 kernel LFP approximation"""
import numpy as np
from numpy.typing import ArrayLike
import scipy
import pickle
import importlib.resources as pkg_resources


def _load_uLFP_A0_profile(fname):
    with pkg_resources.open_binary(__package__, fname) as f:
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
        is_excitatory: ArrayLike,
        elec_coords_mm: ArrayLike = [[0, 0, 0]],
        params: dict = params2020,
    ) -> None:
        """Constructor: caches per-spike contributions to LFP for given neurons.

        Parameters
        ----------
        xs_mm : npt.ArrayLike
            Sequence of length N_n, contains X coordinates of N_n neurons in mm
        ys_mm : npt.ArrayLike
            Sequence of length N_n, contains Y coordinates of N_n neurons in mm
        zs_mm : npt.ArrayLike
            Sequence of length N_n, contains Z coordinates of N_n neurons in mm
        is_excitatory : npt.ArrayLike
            Sequence of length N_n, contains cell type of N_n neurons where
            False (0) represents inhibitory and True (1) represents excitatory
        elec_coords_mm : npt.ArrayLike, optional
            Shape (N_e, 3), where N_e is the number of recording sites and the
            three columns represent X, Y, and Z coordinates.
            By default [[0, 0, 0]]
        params : dict, optional
            Dict containing parameters. See the default params2020 object for
            required elements
        """

        assert len(xs_mm) == len(ys_mm) == len(zs_mm)
        n_neurons = len(xs_mm)
        assert len(is_excitatory) in [1, n_neurons]
        if type(elec_coords_mm) is not np.ndarray:
            elec_coords_mm = np.array(elec_coords_mm)
        assert elec_coords_mm.shape[1] == 3

        ### calc ampltiude and delay for each neuron for each contact
        n_elec = elec_coords_mm.shape[0]
        dist = np.tile(
            elec_coords_mm[:, :2].reshape(n_elec, 2, 1), (1, 1, n_neurons)
        ).astype(
            "float64"
        )  # n_elec X 2 X n_neurons
        dist[:, 0, :] -= xs_mm
        dist[:, 1, :] -= ys_mm
        dist = np.sqrt(np.sum(dist ** 2, axis=1))  # n_elec X n_neurons
        # dist in mm, va in m/s, so dist/va will be in ms
        self._delay = params["d_ms"] + dist / params["va_m_s"]
        # self._delay is n_elec X n_neurons

        # amplitude and width of kernel depend on cell type
        A0 = np.zeros(((n_elec, n_neurons)))
        #     need 2:3 index so it remains a column ⬇
        depths = zs_mm - np.tile(elec_coords_mm[:, 2:3], (1, n_neurons))
        # 2 sigma squared, used in Gaussian kernel
        self._ss = np.ones(n_neurons)
        if len(is_excitatory) == 1 and is_excitatory:
            A0 = params["exc_A0_by_depth"](depths)
            self._ss *= 2 * params["sig_e_ms"] ** 2
        elif len(is_excitatory) == 1 and not is_excitatory:
            A0 = params["inh_A0_by_depth"](depths)
            self._ss *= 2 * params["sig_i_ms"] ** 2
        else:
            A0[:, is_excitatory] = params["exc_A0_by_depth"](depths[:, is_excitatory])
            A0[:, ~is_excitatory] = params["inh_A0_by_depth"](depths[:, ~is_excitatory])
            self._ss[is_excitatory] = 2 * params["sig_e_ms"] ** 2
            self._ss[~is_excitatory] = 2 * params["sig_i_ms"] ** 2

        self._amp = A0 * np.exp(-dist / params["lambda_mm"])
        # self._amp is also n_elec X n_neurons
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
        contribs = amp * np.exp(-(t ** 2) / ss)
        # sum over spikes and return. should be n_eval X n_elec
        lfp = np.sum(contribs, axis=2)
        assert lfp.shape == (len(t_eval_ms), n_elec)
        return lfp
