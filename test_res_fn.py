'''
This file is used to test the reciprocal space resolution function

Created on 2023-09-12 9:33 PM

Author: yfwang09

'''

import numpy as np

res_dict = {
    'zeta_v_rms' : 0.53e-3/2.35,     # incoming divergence in vertical direction, in rad
    'zeta_h_rms': 1e-5/2.35,         # incoming divergence in horizontal direction, in rad 
    'NA_rms' : 7.31e-4/2.35,    # NA of objective, in rad
    'eps_rms' : 0.00006,        # rms width of x-ray energy bandwidth (2.355 factor fwhm -> rms)
    'zl_rms' : 0.6e-6/2.35,     # rms width of the Gaussian beam profile, in m
    'two_theta' : 20.73,        # 2theta in degrees
    'D' : 2*np.sqrt(5e-5*1e-3), # physical aperture of objective, in m
    'd1' : 0.274,               # sample-objective distance, in m
    'TwoDeltaTheta' : 0,        # rotation of the 2 theta arm
    'phi' : -.000, #in rad; sample tilt angle 1
    'chi' : 0, #.015*np.pi/180, #in rad, sample tilt angle 2
    'hkl' : [0, 0, 1], # Miller indices for the diffraction plane
    'ns' : [1, 1, -1], # normal vector of the slip plane
    'bs' : [1, -1, 0], # Burgers vector of the slip system
    'ts' : [1, 1, 2],  # dislocation line direction
    'b' : 1, # 0.3057, # Burger's magnitude (unitless)
    'nu' : 0.334, # Poisson's ratio
    }

res_dict['mu'] = np.deg2rad(res_dict['two_theta'])      # in rad; base tilt
res_dict['theta'] = np.deg2rad(res_dict['two_theta']/2) # in rad
print('physical aperture (rad):', res_dict['D']/res_dict['d1'])