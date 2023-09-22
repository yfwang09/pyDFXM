'''
Classes for forward model and resolution function

Refactored on Sep 18 2023

@author: yfwang09

General form of the DFXM forward model
'''

import time
import numpy as np
from scipy.stats import gaussian_kde, truncnorm
from sympy import N

default_Nrays = 10000000
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
    'b': 2.86e-10,              # Burger's vector magnitude (scaling)
    # Parameters for reciprocal space resolution function
    'Nrays': default_Nrays,
    'q1_range': default_qrange, 'q2_range': default_qrange, 'q3_range': default_qrange, 
    'npoints1': default_ngrids, 'npoints2': default_ngrids, 'npoints3': default_ngrids,
}

default_forward_dict['mu'] = np.deg2rad(default_forward_dict['two_theta'])      # in rad; base tilt
default_forward_dict['theta'] = np.deg2rad(default_forward_dict['two_theta']/2) # in rad

class DFXM_forward():
    def __init__(self, d=default_forward_dict):
        self.d = d          # Input dictionary

    def res_fn(self, saved_q=None, timeit=False):
        ''' Compute the resolution function for DFXM
            The objective is modelled as an isotropic Gaussian with an NA and in addition a square phyical aperture of d side length D. 
            
            Yifan Wang, Sep 21, 2023, version 2

        Parameters
        ----------
        saved_q : tuple, default None
            if not None, the qvectors will be loaded from saved_q
        timeit : bool, default False
            if True, print the time for each step

        Returns
        -------
        Res_qi : array of (Npixels, Npixels, Npixels)
            3D voxelized field of resolution function
            if saved_q is None, the qvectors will be returned as well
        ratio_outside : float
            ratio of rays outside the physical aperture
        '''

        # INPUT instrumental settings
        if timeit:
            tic = time.time()

        d = self.d
        Nrays = d['Nrays']
        q1_range, q2_range, q3_range = d['q1_range'], d['q2_range'], d['q3_range']
        npoints1, npoints2, npoints3 = d['npoints1'], d['npoints2'], d['npoints3']
        d['theta'] = np.deg2rad(d['two_theta']/2)        # half scattering angle, in rad
        phys_aper = d['D']/d['d1']          # physical aperture of objective, in rad

        ######## Ray tracing in crystal system ########
        # Eq. 43-45 in the Poulsen et al. (2021)

        # Sample incoming rays
        if saved_q is None:
            # zeta_v = np.random.randn(Nrays)*d['zeta_v_rms']
            zeta_v = (np.random.rand(Nrays) - 0.5)*d['zeta_v_rms']*2.35 # using uniform distribution to be consistent with Henning's implementation
            zeta_h = np.random.randn(Nrays)*d['zeta_h_rms']
            eps    = np.random.randn(Nrays)*d['eps_rms']

            # Define truncated normal distribution by the physical aperture
            delta_2theta = truncnorm.rvs(-phys_aper/2/d['NA_rms'], phys_aper/2/d['NA_rms'], size=Nrays) * d['NA_rms']
            xi = truncnorm.rvs(-phys_aper/2/d['NA_rms'], phys_aper/2/d['NA_rms'], size=Nrays) * d['NA_rms']

            if timeit:
                print('Time for sampling rays: {:.2f} s'.format(time.time()-tic))

            # Compute q_{rock,roll,par}, the phi, chi shifts are not included for resolution function
            qrock = -zeta_v/2 - delta_2theta/2 # + d['phi']
            qroll = -zeta_h/(2*np.sin(d['theta'])) - xi/(2*np.sin(d['theta'])) # + d['chi']
            qpar = eps + (1/np.tan(d['theta']))*(-zeta_v/2 + delta_2theta/2)

            # Convert from crystal to imaging system
            qrock_prime = np.cos(d['theta'])*qrock + np.sin(d['theta'])*qpar
            q2theta = - np.sin(d['theta'])*qrock + np.cos(d['theta'])*qpar

        else:
            qrock_prime, qroll, q2theta = saved_q

        if timeit:
            print('Time for computing q: {:.2f} s'.format(time.time()-tic))

        # Convert point cloud into local density function, Resq_i, normalized to 1
        # If the range is set too narrow such that some points fall outside ranges,
        #             the fraction of points outside is returned as ratio_outside
        if saved_q is None:
            index1 = np.floor((qrock_prime + q1_range / 2) / q1_range * (npoints1 - 1)).astype(int)
            index2 = np.floor((qroll + q2_range / 2) / q2_range * (npoints2 - 1)).astype(int)
            index3 = np.floor((q2theta + q3_range / 2) / q3_range * (npoints3 - 1)).astype(int)
        else:
            a1 = 1 / q1_range * (npoints1 - 1)
            b1 = (q1_range / 2 + np.cos(d['theta'])*d['phi']) * a1
            a2 = 1 / q2_range * (npoints2 - 1)
            b2 = (q2_range / 2 + d['chi']) * a2
            a3 = 1 / q3_range * (npoints3 - 1)
            b3 = (q3_range / 2 - np.sin(d['theta'])*d['phi']) * a3
            index1 = np.floor(qrock_prime * a1 + b1).astype(int)
            index2 = np.floor(qroll * a2 + b2).astype(int)
            index3 = np.floor(q2theta * a3 + b3).astype(int)

        # count the total number of outside rays
        outside_ind = ((index1 < 0) | (index1 >= npoints1) |
                    (index2 < 0) | (index2 >= npoints2) |
                    (index3 < 0) | (index3 >= npoints3))
        outside = np.count_nonzero(outside_ind)

        # count the histogram of the 3d voxelized space to estimate Resq_i
        ind = np.stack([index1, index2, index3], axis=-1) # (Nrays, 3)
        ind_inside = ind[np.logical_not(outside_ind), :]  # remove the outside voxels
        # print(ind_inside)

        if timeit:
            print('Time for Monte Carlo ray tracing: {:.2f} s'.format(time.time() - tic))

        # count the elements in each voxel, pad to the shape of [np1 np2 np3]
        Res_qi = np.bincount(np.ravel_multi_index(ind_inside.T, (npoints1, npoints2, npoints3)), minlength=npoints1*npoints2*npoints3).reshape((npoints1, npoints2, npoints3)).astype(float)
        Res_qi = Res_qi / np.max(Res_qi) # normalize to 1
        
        ratio_outside = outside / Nrays

        if timeit:
            print('Time for voxelizing: {:.2f} s'.format(time.time() - tic))

        if saved_q is None:
            return Res_qi, (qrock_prime, qroll, q2theta)
        else:
            return Res_qi, ratio_outside

    def get_rot_matrices(self, chi=None, phi=None, mu=None):
        '''Returns rotation matrices reuired to go from sample to lab frame

        Parameters
        ----------
        chi : float, optional
            rolling angle of the sample (rad)
        phi : float, optional
            rocking angle of the sample (rad)
        mu : float, optional
            base tilt of the sample (rad)

        Returns
        -------
        Chi : numpy array
            rotation matrix for rolling
        Phi : numpy array
            rotation matrix for rocking
        Mu : numpy array
            rotation matrix for base tilt
        '''
        if chi is None:
            chi = self.d['chi']
        if phi is None:
            phi = self.d['phi']
        if mu is None:
            mu = self.d['mu']
        Chi = np.array([[1,           0,            0], 
                        [0, np.cos(chi), -np.sin(chi)],
                        [0, np.sin(chi),  np.cos(chi)],
                       ])
        Phi = np.array([[ np.cos(phi), 0, np.sin(phi)],
                        [           0, 1,           0],
                        [-np.sin(phi), 0, np.cos(phi)],
                       ])
        Mu = np.array([[ np.cos(mu), 0, np.sin(mu)],
                       [         0,  1,          0],
                       [-np.sin(mu), 0, np.cos(mu)],
                      ])
        return Chi, Phi, Mu