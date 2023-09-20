'''
Classes for forward model and resolution function

Refactored on Sep 18 2023

@author: yfwang09

General form of the DFXM forward model
'''

import numpy as np

default_qrange = 8e-3
default_ngrids = 40

default_forward_dict = {
    'psize' : 75e-9,            # pixel size (m)
    # Setup of the Optics (default from Dresselhaus-Marais et al., 2021)
    # Note: 2.355 factor if converting fwhm -> rms
    'zeta_v_rms' : 0.53e-3/2.35,# incident beam vertical variance (rad)
    'zeta_h_rms': 1e-5/2.35,    # incident beam horizontal variance (rad) 
    'NA_rms' : 7.31e-4/2.35,    # Numerical Aperture variance (rad)
    'eps_rms' : 0.00006,        # incident x-ray energy variance (eV)
    'zl_rms' : 0.6e-6/2.35,     # Gaussian beam width variance (m)
    'two_theta' : 20.73,        # 2theta for Al-(002) (deg)
    'D' : 2*np.sqrt(5e-5*1e-3), # physical aperture of objective (m)
    'd1' : 0.274,               # sample-objective distance (m)
    # Setup of the Ghoniometer
    'TwoDeltaTheta' : 0,        # additional rotation of the 2theta arm
    'phi' : -.000,              # in rad, sample tilt angle 1 (rocking)
    'chi' : 0,                  # in rad, sample tilt angle 2 (rolling)
    'omega' : 0,                # in rad, sample rotation (in-plane)
    # Parameters for reciprocal space resolution function
    'Nrays': 10000000,
    'q1_range': default_qrange, 'q2_range': default_qrange, 'q3_range': default_qrange, 
    'npoints1': default_ngrids, 'npoints2': default_ngrids, 'npoints3': default_ngrids,
    # Setup of the Crystal system
    # 'hkl' : [2, -2, 0], # Miller indices for the diffraction plane (this is zlab for the crystal orientation)
    # 'xlab': [0, 0, 2], # Crystal orientation of the x direction in the lab system assuming lab and grain are the same
    # 'ylab': [-2, -2, 0], #Crystal orientation of the y direction in the lab system assuming lab and grain are the same
    # 'ns' : [1, 1, -1], # normal vector of the slip plane
    # 'bs' : [1, -1, 0], # Burgers vector of the slip system
    # 'ts' : [1, 1, 2],  # dislocation line direction
    # 'b' : -2.86e-10, # 0.3057, # Burger's magnitude (unitless)
    # 'nu' : 0.3, # Poisson's ratio
    # Parameters for resolution function calculation
    }

default_forward_dict['mu'] = np.deg2rad(default_forward_dict['two_theta'])      # in rad; base tilt
default_forward_dict['theta'] = np.deg2rad(default_forward_dict['two_theta']/2) # in rad

class DFXM_forward():
    def __init__(self, d=default_forward_dict):
        self.d = d          # Input dictionary