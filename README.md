# pyDFXM: A Python simulator for dark-field X-ray microscopy (DFXM)

This is a Python simulator for dark-field X-ray microscopy (DFXM). It is based on the MATLAB implementation of the formulation in the [Poulsen et al. (2021) paper](https://scripts.iucr.org/cgi-bin/paper?S1600576721007287) and [Poulsen et al. (2017) paper](https://scripts.iucr.org/cgi-bin/paper?S1600576717011037).

References:
1. H. F. Poulsen, A. C. Jakobsen, H. Simons, S. R. Ahl, P. K. Cook, and C. Detlefs, X-Ray Diffraction Microscopy Based on Refractive Optics, J Appl Crystallogr 50, 1441 (2017).
2. H. F. Poulsen, L. E. Dresselhaus-Marais, M. A. Carlsen, C. Detlefs, and G. Winther, Geometrical-Optics Formalism to Model Contrast in Dark-Field X-Ray Microscopy, J Appl Crystallogr 54, 1555 (2021).

## Installation

The simulator requires the following Python packages:

- `numpy`
- `scipy`
- `matplotlib`

## Code structure

The code (Supplementary material of the Poulsen et al., 2021 paper) can be separated into three distinct parts:
* The resolution function (res_fxn_q)
* Displacement gradient of edge dislocation (edge_disl_helper)
* DFXM forward model (forward_model_setup)

We will test them separately in the following example cases and write necessary notes to help the users to better understand how to use and modify the code.

## Example cases

The following example cases are provided:

- `test_resolution.py`: Test the resolution function of the simulator.
- `test_edge_disl.py`: Simulate the DFXM image of a single straight edge dislocation.