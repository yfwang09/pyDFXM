'''
Classes for forward model and resolution function

Refactored on Sep 18 2023

@author: yfwang09

General form of the DFXM forward model
'''

import time
import os
import numpy as np
from scipy.stats import gaussian_kde, truncnorm
import matplotlib.pyplot as plt

def default_forward_dict():
    '''Generate a default forward model dictionary'''
    default_Nrays = 10000000
    default_qrange = 8e-3
    default_ngrids = 40

    forward_dict = {
        ################## Material and Grain setup #######################
        # Setup of the sample coordinate system (Ug, Eq. 7-8)
        # Note by Yifan: 2023-12-07
        # The sample coordinate system is always dealt with in the
        # forward_model class. In dispgrad_func class, the grain system
        # is always aligned with identity matrix in Miller indices
        'x_c': [1,0,0],             # x dir. for the crystal system (Fig.2)
        'y_c': [0,1,0],             # y dir. for the crystal system (Fig.2)
        'hkl': [0,0,1],             # hkl diffraction plane, z dir. crystal
        ## 'Ug': np.identity(3),    # or directly define a rotation matrix
        'two_theta' : 20.73,        # 2theta for Al-(002) (deg)
        'b': 2.86e-10,              # Burger's vector magnitude (scaling, not used in forward model)
        ###################################################################
        # Parameters for the forward model
        'psize' : 75e-9,            # pixel size (m)
        'Npixels' : [50, 45, 40],   # number of pixels in a half space
        ###################################################################
        # Parameters for reciprocal space resolution function
        'Nrays': default_Nrays,
        'q1_range': default_qrange, 'npoints1': default_ngrids,
        'q2_range': default_qrange, 'npoints2': default_ngrids,
        'q3_range': default_qrange, 'npoints3': default_ngrids,
        ###################################################################
        # Setup of the Optics (default from Dresselhaus-Marais et al., 2021)
        # Note: 2.355 factor if converting fwhm -> rms
        'zeta_v_rms' : 0.53e-3/2.35,# incident beam vertical variance (rad)
        'zeta_h_rms': 1e-5/2.35,    # incident beam horizontal variance (rad) 
        'NA_rms' : 7.31e-4/2.35,    # Numerical Aperture variance (rad)
        'eps_rms' : 0.00006,        # incident x-ray energy variance (eV)
        'zl_rms' : 0.6e-6/2.35,     # Gaussian beam width variance (m)
        'D' : 2*np.sqrt(5e-5*1e-3), # physical aperture of objective (m)
        'd1' : 0.274,               # sample-objective distance (m)
        # Setup of the Ghoniometer
        'TwoDeltaTheta' : 0,        # additional rotation of the 2theta arm
        'phi' : 0,                  # in rad, sample tilt angle 1 (rocking)
        'chi' : 0,                  # in rad, sample tilt angle 2 (rolling)
        'omega' : 0,                # in rad, sample rotation (in-plane)
    }
    forward_dict['mu'] = np.deg2rad(forward_dict['two_theta'])      # in rad
    forward_dict['theta'] = np.deg2rad(forward_dict['two_theta']/2) # in rad
    return forward_dict

class DFXM_forward():
    def __init__(self, d=default_forward_dict, load_res_fn=None, verbose=False):
        # get the grain rotation matrix
        if 'Ug' not in d.keys():
            if 'hkl' in d.keys():
                z_c = d['hkl']/np.linalg.norm(d['hkl'])
                if 'x_c' in d.keys():
                    x_c = d['x_c']/np.linalg.norm(d['x_c'])
                else:
                    x_c = [1, 0, 0]
                if 'y_c' in d.keys():
                    y_c = d['y_c']/np.linalg.norm(d['y_c'])
                else:
                    y_c = np.cross(z_c, x_c)
                    x_c = np.cross(y_c, z_c)
                # For definition of Ug = U in Eq. 5, U = [x_c^T, y_c^T, z_c^T]^T
                # d['Ug'] = np.transpose([x_c, y_c, z_c])
                d['Ug'] = np.array([x_c, y_c, z_c])
            else:
                d['Ug'] = np.eye(3)

        # Reset the Goniometer
        if 'phi' not in d.keys(): d['phi'] = 0
        if 'chi' not in d.keys(): d['chi'] = 0
        if 'omega' not in d.keys(): d['omega'] = 0

        self.d = d          # Input dictionary
        self.Ug = d['Ug']

        if load_res_fn is not None:
            if os.path.exists(load_res_fn):
                self.Res_qi = np.load(load_res_fn)['Res_qi']
            else:
                self.Res_qi, _ = self.res_fn(timeit=verbose)
                np.savez_compressed(load_res_fn, Res_qi=self.Res_qi)
        else:
            print('=========================================================================')
            print('Resolution function is not calculated, DO NOT use the forward function')
            print('Pass a filename load_res_fn to create/load the resolution function')
            print('=========================================================================')

    def res_fn(self, saved_q=None, plot=False, timeit=False):
        ''' Compute the resolution function for DFXM
            The objective is modelled as an isotropic Gaussian with an NA and in addition a square phyical aperture of d side length D. 
            
            Yifan Wang, Sep 21, 2023, version 2

        Parameters
        ----------
        saved_q : tuple, default None
            if not None, the qvectors will be loaded from saved_q
        plot : bool, default False
            whether to plot the Monte Carlo ray-tracing of the resolution function
        timeit : bool, default False
            if True, print the time for each step

        Returns
        -------
        Res_qi : array of (Npixels, Npixels, Npixels)
            3D voxelized field of resolution function
            if saved_q is None, the qvectors will be returned as well
        ratio_outside : float
            ratio of rays outside the physical aperture, will only be returned when saved_q is None
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
            zeta_v = np.random.randn(Nrays)*d['zeta_v_rms']
            # zeta_v = (np.random.rand(Nrays) - 0.5)*d['zeta_v_rms']*2.35 # using uniform distribution to be consistent with Henning's implementation
            zeta_h = np.random.randn(Nrays)*d['zeta_h_rms']
            eps    = np.random.randn(Nrays)*d['eps_rms']

            # Define truncated normal distribution by the physical aperture
            delta_2theta = truncnorm.rvs(-phys_aper/2/d['NA_rms'], phys_aper/2/d['NA_rms'], size=Nrays) * d['NA_rms']
            xi = truncnorm.rvs(-phys_aper/2/d['NA_rms'], phys_aper/2/d['NA_rms'], size=Nrays) * d['NA_rms']

            if timeit:
                print('Time for sampling rays: {:.2f} s'.format(time.time()-tic))

            # Compute q_{rock,roll,par},
            # phi&chi shifts are for testing only - NEVER change them when calculating res_fn!
            qrock = -zeta_v/2 - delta_2theta/2 + d['phi']
            qroll = -zeta_h/(2*np.sin(d['theta'])) - xi/(2*np.sin(d['theta'])) + d['chi']
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

        if plot:
            plot_half_range = 0.0045
            _, ax = self.plot_res(qrock, qroll, qpar, plot_half_range=plot_half_range, show=False)
            ax.set_xlabel(r'$\hat{q}_{rock}$')
            ax.set_ylabel(r'$\hat{q}_{roll}$')
            ax.set_zlabel(r'$\hat{q}_{par}$')
            ax.set_title('Crystal system')
            plt.show()
            _, ax = self.plot_res(qrock_prime, qroll, q2theta, plot_half_range=plot_half_range, show=False)
            ax.set_xlabel(r'$\hat{q}_{rock}^\prime$')
            ax.set_ylabel(r'$\hat{q}_{roll}$')
            ax.set_zlabel(r'$\hat{q}_{2\theta}$')
            ax.set_title('Imaging system')
            plt.show()
            
        if saved_q is None:
            return Res_qi, (qrock_prime, qroll, q2theta)
        else:
            return Res_qi, ratio_outside

    def plot_res(self, qrock, qroll, qpar, plot_half_range=4.5e-3, show=True):
        Nrays = len(qrock)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        Nskip = Nrays//10000 + 1
        
        qrock = qrock[::Nskip]
        qroll = qroll[::Nskip]
        qpar = qpar[::Nskip]
        ax.plot(qrock, qroll, qpar, 'ok', markersize=1)
        ax.plot(np.zeros_like(qrock)-plot_half_range, qroll, qpar, 'o', markersize=1)
        ax.plot(qrock, np.zeros_like(qroll)+plot_half_range, qpar, 'o', markersize=1)
        ax.plot(qrock, qroll, np.zeros_like(qpar)-plot_half_range, 'o', markersize=1)
        ax.set_xlim([-plot_half_range, plot_half_range])
        ax.set_ylim([-plot_half_range, plot_half_range])
        ax.set_zlim([-plot_half_range, plot_half_range])
        if show:
            plt.show()
        else:
            return fig, ax

    def forward(self, Fg_fun, Res_qi=None, timeit=False):
        '''Generate voxelized intensities and the image for DFXM

        Parameters
        ----------
        d : dict
            dictionary for dislocation and instrumental settings
        Fg_fun : function handle
            function for calculating the displacement gradient tensor
        Res_qi : array, default None
            pre-computed Res_qi, if None, the Res_qi is calculated
        timeit : bool
            whether print the timing info of the algorithm

        Returns
        -------
        im : array of (Npixels, Npixels)
            image of DFXM given the strain tensor
        qi_field : array of (Npixels, Npixels, Npixels, 3)
            3D voxelized field of intensities
        '''
        d = self.d
        if type(d['Npixels']) is int:
            Nx = Ny = Nz = d['Npixels']
        else:
            Nx, Ny, Nz = d['Npixels']
        Nsub = 2                # multiply 2 to avoid sampling the 0 point, make the grids symmetric over 0
        NNx, NNy, NNz = Nsub*Nx, Nsub*Ny, Nsub*Nz
        # NN = Nsub*Npixels     # NN^3 is the total number of "rays" (voxels?) probed in the sample

        # INPUT instrumental settings, related to direct space resolution function
        psize = d['psize']   # pixel size in units of m, in the object plane
        zl_rms = d['zl_rms'] # rms value of Gaussian beam profile, in m, centered at 0
        theta_0 = np.deg2rad(d['two_theta']/2) # in rad
        v_hkl = d['hkl']
        TwoDeltaTheta = d['TwoDeltaTheta']
        U = d['Ug']
        phi = d['phi']
        chi = d['chi']
        omega = d['omega']

        if timeit: 
            tic = time.time()

        # Obtain the resolution function (calculated when initializing the class)
        if Res_qi is None:
            Res_qi = self.Res_qi

        # Define the grid of points in the lab system (xl, yl, zl)
        theta = theta_0 + TwoDeltaTheta
        yl_start = -psize*Ny/2 + psize/(2*Nsub) # start in yl direction, in units of m, centered at 0
        yl_step = psize/Nsub
        xl_start = ( -psize*Nx/2 + psize/(2*Nsub) )/np.tan(2*theta) # start in xl direction, in m, for zl=0
        xl_step = psize/Nsub/np.tan(2*theta)
        zl_start = -0.5*zl_rms*6 # start in zl direction, in m, for zl=0
        zl_step = zl_rms*6/(NNz-1)

        qi1_start, qi1_step = -d['q1_range']/2, d['q1_range']/(d['npoints1']-1)
        qi2_start, qi2_step = -d['q2_range']/2, d['q2_range']/(d['npoints2']-1)
        qi3_start, qi3_step = -d['q3_range']/2, d['q3_range']/(d['npoints3']-1)

        Q_norm = np.linalg.norm(v_hkl)  # We have assumed B_0 = I (?)
        q_hkl = v_hkl/Q_norm

        # Define the rotation matrices
        mu = theta_0
        M = [[np.cos(mu), 0, np.sin(mu)],
            [0, 1, 0],
            [-np.sin(mu), 0, np.cos(mu)],
        ]
        Omega = np.eye(3)
        Chi = np.eye(3)
        Phi = np.eye(3)
        Gamma = M@Omega@Chi@Phi
        Theta = [[np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
        ]

        im = np.zeros([Nx,Ny])  # The forward model image
        # qi_field = np.zeros([NNx,NNy,NNz,3]) # wave vector function
        
        # The tensor fields Fg, Hg and the vector fields qs, qc, qi are all defined as 3D fields in the lab system

        yl = yl_start + np.arange(NNy)*yl_step
        zl = zl_start + np.arange(NNz)*zl_step
        xl0= xl_start + np.arange(NNx)*xl_step
        # rulers = np.array([xl0, yl, zl]) # rulers in the lab system (for plotting)
        rulers = (xl0, yl, zl)
        # create the 3D grid of points in the lab system, the first dimension is zl, the second is yl, the third is xl
        # YL[:,i,j] == yl; ZL[i,:,j] == zl; XL0[i,j,:] == xl0;
        ZL, YL, XL0 = np.meshgrid(zl, yl, xl0)

        if timeit:
            print('Initialization time: %.2fs'%(time.time() - tic))
        
        XL = XL0 + ZL/np.tan(2*theta)           # Project to image system
        PZ = np.exp(-0.5*(ZL/zl_rms)**2)        # Gaussian beam in zl (a thin slice of sample)
        RL = np.stack([XL, YL, ZL], axis=-1)    # (NNy,NNz,NNx,3)
        # Determine the location of the pixel on the detector
        DET_IND_Y = np.round((YL-yl_start)/yl_step).astype(int)//Nsub # THIS ALIGNS WITH yl
        DET_IND_Z = np.round((XL0-xl_start)/xl_step).astype(int)//Nsub # THIS IS THE OTHER DETECTOR DIRECTION AND FOLLOWS xl BUT WITH MAGNIFICATION
        #### This line is WRONG, the rotation is reversed ####
        # RS = np.einsum('ji,...j->...i', Gamma, RL) # NB: Gamma inverse Eq. 5
        #### The correct lines are ####
        RS = np.einsum('ij,...j->...i', Gamma, RL) # Eq. 5
        #########################################################
        RG = np.einsum('ji,...j->...i', U, RS)     # NB U inverse, Eq. 7
        Fg = Fg_fun(RG[..., 0], RG[..., 1], RG[..., 2]) # calculate the displacement gradient

        # determine the qi for given voxel
        Hg = np.swapaxes(np.linalg.inv(Fg), -1, -2) - np.eye(3) # Eq. 31
        QS = np.einsum('ij,...jk,k->...i', U, Hg, q_hkl)        # Eq. 32
        QC = QS + np.array([phi - TwoDeltaTheta/2, chi, (TwoDeltaTheta/2)/np.tan(theta_0)])                                          # Eq. 40 (also Eq. 20)
        QI = np.einsum('ij,...j->...i', Theta, QC)              # Eq. 41
        qi_field = np.swapaxes(np.swapaxes(QI, 2, 1), 1, 0)     # for plotting, sorted in order x_l,y_l,z_l,:

        # Interpolation in rec. space resolution function.
        IND1 = np.floor( (QI[...,0] - qi1_start)/qi1_step).astype(int)
        IND2 = np.floor( (QI[...,1] - qi2_start)/qi2_step).astype(int)
        IND3 = np.floor( (QI[...,2] - qi3_start)/qi3_step).astype(int)

        if timeit:
            print('Calculate the wave vectors: %.2fs'%(time.time() - tic))

        # Determine intensity contribution from voxel based on rec.space res.function
        PROB = np.zeros_like(PZ)
        IND_IN = ((IND1 >= 0) & (IND1 < d['npoints1']) &
                  (IND2 >= 0) & (IND2 < d['npoints2']) &
                  (IND3 >= 0) & (IND3 < d['npoints3'])
        )
        # for i,j,k in zip(np.nonzero(IND_IN)[0], np.nonzero(IND_IN)[1], np.nonzero(IND_IN)[2]):
        #     PROB[i,j,k] = Res_qi[IND1[i,j,k],IND2[i,j,k],IND3[i,j,k]]*PZ[i,j,k]
        PROB[IND_IN] = Res_qi[IND1[IND_IN],IND2[IND_IN],IND3[IND_IN]]*PZ[IND_IN]

        # Sum over all pixels in the detector, equivalent to the following loops but faster
        # for i in range(NN):
        #     for j in range(NN):
        #         for k in range(NN):
        #             # im[k//Nsub, i//Nsub] += PROB[i,j,k]
        #             im[DET_IND_Z[i,j,k], DET_IND_Y[i,j,k]] += PROB[i,j,k]
        ravel_ind = np.ravel_multi_index((DET_IND_Z.flatten(), DET_IND_Y.flatten()), (Nx, Ny)) # shape (NNx*NNy*NNz,), values in range(Nx*Ny)
        im = np.bincount(ravel_ind.flatten(), weights=PROB.flatten(), minlength=Nx*Ny).reshape(Nx,Ny) # shape (Nx,Ny)

        if timeit:
            print('Image calculation: %.2fs'%(time.time() - tic))

        return im, qi_field, rulers

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
        Mu = np.array([[ np.cos(mu), 0, -np.sin(mu)],
                       [         0,  1,          0],
                       [ np.sin(mu), 0,  np.cos(mu)],
                      ])            # Eq. 14, opposite to phi
        return Chi, Phi, Mu