from logging import raiseExceptions
import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d
from numpy.linalg import norm
import pickle
import importlib.resources as pkg_resources

class WSLFP:
    def __init__(self, xs, ys, zs, elec_coords, alpha=1.65, tau_ampa_ms=6, tau_gaba_ms=0):
        """xs, ys, zs are n_neurons-length 1D arrays
        elec_coords is a (n_elec, 3) array
        [[1, 1, 1],
         [2, 2, 2]] """
        self.a = self._amplitude(xs, ys, zs, elec_coords)
        self.alpha = alpha
        self.tau_ampa_ms = tau_ampa_ms
        self.tau_gaba_ms = tau_gaba_ms

    def _amplitude(self, xs, ys, zs, elec_coords):
        ...

    def compute(ampa: np.ndarray, t_ampa_ms, gaba: np.ndarray, t_gaba_ms, t_eval_ms: np.ndarray):
        """_summary_
​
        Parameters
        ----------
        ampa : np.ndarray
            (n_timepoints, n_neurons)  e.g., 5 timepoints * 1000 neurons => 5000
        t_ampa_ms : np.ndarray
            (n_timepoints), e.g., [1, 2, 3, 4, 5]
        gaba : np.ndarray
            (n_timepoints, n_neurons)
        t_gaba_ms : _type_
            _description_
        t_eval_ms : _type_
            _description_
​
        Example
        -------
        # just 1 timepoint
        lfp = wslfp.compute(..., t_gaba_ms=[now_ms], t_eval_ms=[now_ms])
        # multiple timepoint
        lfp = wslfp.compute(..., t_gaba_ms=[multiple, gaba, measurements], t_eval_ms=[a, whole, bunch, of, timepoints])
        """

        try:
            _check_timepoints(t_ampa_ms, t_gaba_ms, t_eval_ms)


def compute_ampa_time(t_ampa_ms, tau_ampa):
    ampa_time_arr = []
    for x in range(len(t_ampa_ms)):
        ampa_time_arr[x] = t_ampa_ms[x] - tau_ampa
    return ampa_time_arr

# use numpy array instead of loop

def compute_gaba_time(t_gaba_ms, tau_gaba):
    gaba_time_arr = []
    for x in range(len(t_gaba_ms)):
       gaba_time_arr[x] = t_gaba_ms[x] - tau_gaba
    return gaba_time_arr

def compute_ampa(ampa:np.ndarray, t_ampa_ms, tau_ampa):
    ampa_arr = []
    time = compute_ampa_time(t_ampa_ms, tau_ampa)
    for x in range(len(time)):
        ampa_arr[x] = ampa[x] * time[x]
    return ampa_arr

def compute_gaba(gaba:np.ndarray, t_gaba_ms, tau_gaba):
    gaba_arr = []
    time = compute_gaba_time(t_gaba_ms, tau_gaba)
    for x in range(len(time)): #no for loop
        gaba_arr[x] = gaba[x] * time[x] #should be gaba at that time
    return gaba_arr

def sum_ampa(ampa, t_ampa_ms, tau_ampa):
    ampa_sum = 0
    ampa_arr = compute_ampa(ampa, t_ampa_ms, tau_ampa)
    for ampa_curr in ampa_arr:
        ampa_sum += ampa_curr
    return ampa_sum

def sum_gaba(gaba, t_gaba_ms, tau_gaba):
    gaba_sum = 0
    gaba_arr = compute_gaba(gaba, t_gaba_ms, tau_gaba)
    for gaba_curr in gaba_arr:
        gaba_sum += gaba_curr
    return gaba_sum

def _check_timepoints(t_ampa_ms, t_gaba_ms, tau_ampa, tau_gaba, t_eval_ms):
        # need exact timepoints if just one measurement is given. Otherwise, let interpolation throw an error
        # when out of range
        # check t_ampa_ms: ranging from at least tau_ampa ms before the first eval point
        # up to 6 ms before the last eval point

        # check if gaba is at least 6 ms before eval
        # ampa has to be equal to or smaller than eval
        # for a range of ampa, if eval is within range, use linear/quadratic interpolation
        # if not, raise exception
    for t in [np.min(t_eval_ms), np.max(t_eval_ms)]:
            if len(t_ampa_ms) == 1:
                if t > t_ampa_ms[0]:
                    raise Exception("ampa not valid")
            elif t < np.min(t_ampa_ms) or t > np.max(t_gaba_ms):
                raise Exception("ampa not valid")
            #else:
                #t_ampa_chosen = interp1d(t_ampa_ms, t_eval_ms, kind = int)
                #if t > t_ampa_chosen:
                    #raise Exception("ampa not valid")

            if t - tau_gaba < np.min(t_gaba_ms) or t - tau_gaba > np.max(t_gaba_ms):
                raise Exception("gaba not valid")
 
         


        # if not ampa_valid:
        #     raise Exception("ampa not valid")
        # check t_gaba_ms: ranging from tau_gaba before the first eval point
        # up to the last eval point
        # if not gaba_valid:
        #     raise Exception("gaba not valid")

