import numpy as np
import numpy.typing as npt
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
    def __init__(
        self,
        xs_mm,
        ys_mm,
        zs_mm,
        is_excitatory,
        elec_coords_mm=[[0, 0, 0]],
        params=params2020,
    ) -> None:

        assert len(xs_mm) == len(ys_mm) == len(zs_mm)
        n_neurons = len(xs_mm)
        assert len(is_excitatory) in [1, n_neurons]
        if type(elec_coords_mm) is not np.ndarray:
            elec_coords_mm = np.array(elec_coords_mm)

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
            A0[:, ~is_excitatory] = params["inh_A0_by_depth"](
                depths[:, ~is_excitatory]
            )
            self._ss[is_excitatory] = 2 * params["sig_e_ms"] ** 2
            self._ss[~is_excitatory] = 2 * params["sig_i_ms"] ** 2

        self._amp = A0 * np.exp(-dist / params["lambda_mm"])
        # self._amp is also n_elec X n_neurons
        self.params = params

    def compute(
        self,
        i_spikes: npt.ArrayLike,
        t_spikes_ms: npt.ArrayLike,
        t_eval_ms: npt.ArrayLike,
    ):
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
