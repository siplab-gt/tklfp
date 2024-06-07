# tklfp - Teleńczuk Kernel LFP

[![DOI](https://zenodo.org/badge/440986279.svg)](https://zenodo.org/badge/latestdoi/440986279)

This is a lightweight package for computing the kernel LFP approximation from 
[Teleńczuk et al., 2020](https://www.sciencedirect.com/science/article/pii/S0165027020302946).
This method approximates LFP from spikes alone, without the need for more expensive simulations of spatially extended neurons.
See the original authors' demo code [here](https://doi.org/10.5281/zenodo.3866253).

This package was developed by [Kyle Johnsen](https://kjohnsen.org) under the direction of [Chris Rozell](https://siplab.gatech.edu) at Georgia Institute of Technology.

## How to install:
Simply install from pypi:
```bash
pip install tklfp
```

## How to use:

### Initialization
First you must initialize a `TKLFP` object which computes and caches the per-spike contribution of each neuron to the total LFP. You will need X, Y, and Z coordinates of the neurons, their cell types (excitatory/inhibitory, represented as a boolean), and the coordinates of the electrode(s):
```python
from tklfp import TKFLP
tklfp = TKLFP(xs_mm, ys_mm, zs_mm, is_excitatory, elec_coords_mm)
```

The first four arguments must all have the same length N_n, the total number of neurons. `elec_coords_mm` must an N_e by 3 array-like object, where N_e is the number of recording sites.

### Computing LFP
LFP can then be computed with the neuron indices and times of spikes (indices must be between 0 and N_n - 1, corresponding to the parameters given on initialization), as well as the timepoints to evaluate at (must be an iterable):
```python
lfp = tklfp.compute(i_n, t_ms, t_eval_ms)
```

A complete example, reworking the demo from the original paper, can be found [here](https://github.com/siplab-gt/tklfp/blob/master/notebooks/demo_lfp_kernel.ipynb).
Basic usage information is also accessible in docstrings.

### Cortical orientation
The `TKLFP` constructor can also take an `orientation` argument which represents which direction is "up," that is, towards the surface of the cortex or parallel to the apical dendrites of pyramidal cells.
The default is `[0, 0, 1]`, indicating that the positive z axis is "up."
In the case your population isn't a sheet of neurons with uniform orientation (for a curved cortical area, for example), you can pass an N_n by 3 array containing the individual orientation vectors for all the neurons.

## Future development
The package uses [parameters from the original 2020 paper](https://github.com/kjohnsen/tklfp/blob/master/notebooks/param_prep.ipynb) by default. This can be changed by passing in an alternate parameter dictionary on initialization:
```python
tklfp = TKLFP(..., params=new_params)
```

The new params must have the same content as the default [`tklfp.params2020`](https://github.com/kjohnsen/tklfp/blob/master/tklfp/__init__.py#:~:text=_sig_i%20%3D%202.1-,params2020%20%3D,-%7B). The `A0_by_depth` params are scipy interpolation objects, but could theoretically be any callable that will return A0 (in μV) for an arbitrary depth (in mm).

## Citation
Please cite the [publication for the Cleo simulator](https://www.biorxiv.org/content/10.1101/2023.01.27.525963v1) if you use this software in your research.
You may also cite the [Zenodo DOI](https://zenodo.org/badge/latestdoi/440986279) for this repository.