#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Displacement gradient functions

Refactored on Aug 24 2023

@author: yfwang09

General form of the displacement gradient function:
    Fg = Fg_func(d, xg, yg, zg)

Fg_test: pure shear in a sphere
Fg_disl: single edge dislocation
Fg_load: load a displacement gradient field from file (todo)
Fg_disl_network: dislocation network (todo)
"""

import os
import numpy as np
import disl_io_helper as dio
# from disl_io_helper import read_vtk, write_ca
# from io_helper import default_ca_header, simulation_cell_header, cluster_header
import displacement_grad_helper as dgh

def default_dispgrad_dict(dispgrad_type='simple_shear'):
    '''Generate a default displacement gradient dictionary
    
    Args:
        dispgrad_type (str): The type of displacement gradient (strain field).
            Defaults to 'simple_shear'.
    
    Returns:
        dict: The default displacement gradient dictionary.
    
    Raises:
        NotImplementedError: If the specified dispgrad_type is not implemented.
    
    '''
    dispgrad_dict = {
    # Materials properties (For Al by default)
    'b': 2.86e-10, 'nu': 0.334,
    # Grain rotation (Ug, Eq. 7-8)
    # Notes by Yifan: 2023-12-07
    # These are not used since we always align the system with Miller
    # indices. The rotation is always dealt with in the forward_model 
    # class.
    ## 'Ug': np.identity(3), # or directly define a rotation matrix
    # Displacement gradient (strain field) type
    'dispgrad_type': dispgrad_type,
    ## 'dispgrad_type': 'simple_shear', # simple shear in a sphere
    ## 'dispgrad_type': 'edge_disl', # single edge dislocation case
    ## 'dispgrad_type': 'disl_network', # dislocation network (todo)
    }
    if dispgrad_dict['dispgrad_type'] == 'simple_shear':
        dispgrad_dict['R0'] = dispgrad_dict['b']*5000
        dispgrad_dict['components'] = (2, 0)
        dispgrad_dict['strain_magnitude'] = 5e-4
    elif dispgrad_dict['dispgrad_type'] == 'edge_disl':
        dispgrad_dict['bs'] = [1,-1, 0] # Burger's vector dir. in Miller (sample)
        dispgrad_dict['ns'] = [1, 1,-1] # Normal vector dir. in Miller (sample)
        dispgrad_dict['ts'] = [1, 1, 2] # Dislocation line dir. in Miller (sample)
    elif dispgrad_dict['dispgrad_type'] == 'disl_network':
        # print('dispgrad_type == disl_network is not implemented')
        dispgrad_dict['b'] = 1                      # Use Burger's vector as the unit length
        dispgrad_dict['nu'] = 0.334                 # Poisson's ratio (Al)
        dispgrad_dict['a'] = 1.0*dispgrad_dict['b'] # Non-singular radius
    else:
        raise NotImplementedError('dispgrad_type == %s is not implemented'%dispgrad_dict['dispgrad_type'])

    return dispgrad_dict

def return_dis_grain_matrices_all():
    """
    Returns a 3D array containing displacement gradient matrices for all grain orientations.

    Returns:
    dis_grain_all (ndarray): A 3D array of shape (3, 3, 12) containing displacement gradient matrices.
                            Each matrix represents the displacement gradient for a specific grain orientation.
    """
    dis_grain_all = np.zeros([3, 3, 12])
    dis_grain_all[:,:,0] = [[1/np.sqrt(2), 1/np.sqrt(2), 0], [-1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],\
                            [1/np.sqrt(6), -1/np.sqrt(6), 2/np.sqrt(6) ]]
    dis_grain_all[:,:,1] = [[1/np.sqrt(2), 1/np.sqrt(2), 0], [1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)],\
                            [1/np.sqrt(6), -1/np.sqrt(6), -2/np.sqrt(6) ]]
    dis_grain_all[:,:,2] = [[1/np.sqrt(2), -1/np.sqrt(2), 0], [-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)],\
                            [1/np.sqrt(6), 1/np.sqrt(6), -2/np.sqrt(6) ]]
    dis_grain_all[:,:,3] = [[1/np.sqrt(2), -1/np.sqrt(2), 0], [1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)],\
                            [1/np.sqrt(6), 1/np.sqrt(6), 2/np.sqrt(6) ]]
    dis_grain_all[:,:,4] = [[1/np.sqrt(2), 0, -1/np.sqrt(2)], [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],\
                            [1/np.sqrt(6), -2/np.sqrt(6), 1/np.sqrt(6) ]]
    dis_grain_all[:,:,5] = [[1/np.sqrt(2), 0, -1/np.sqrt(2)], [1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)],\
                            [-1/np.sqrt(6), -2/np.sqrt(6), -1/np.sqrt(6) ]]
    dis_grain_all[:,:,6] = [[1/np.sqrt(2), 0, 1/np.sqrt(2)], [1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)],\
                            [-1/np.sqrt(6), 2/np.sqrt(6), 1/np.sqrt(6) ]]
    dis_grain_all[:,:,7] = [[1/np.sqrt(2), 0, 1/np.sqrt(2)], [-1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],\
                            [-1/np.sqrt(6), -2/np.sqrt(6), 1/np.sqrt(6) ]]
    dis_grain_all[:,:,8] = [[0, 1/np.sqrt(2), 1/np.sqrt(2)], [1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)],\
                            [-2/np.sqrt(6), 1/np.sqrt(6), -1/np.sqrt(6) ]]
    dis_grain_all[:,:,9] = [[0, 1/np.sqrt(2), 1/np.sqrt(2)], [-1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)],\
                             [-2/np.sqrt(6), -1/np.sqrt(6), 1/np.sqrt(6) ]]
    dis_grain_all[:,:,10] = [[0, 1/np.sqrt(2), -1/np.sqrt(2)], [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],\
                             [2/np.sqrt(6), -1/np.sqrt(6), -1/np.sqrt(6) ]]
    dis_grain_all[:,:,11] = [[0, 1/np.sqrt(2), -1/np.sqrt(2)], [-1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],\
                             [2/np.sqrt(6), 1/np.sqrt(6), 1/np.sqrt(6) ]]
    return dis_grain_all

def return_dis_grain_matrices(b=None, n=None, t=None):
    ''' Returns the rotation matrix from the dislocation coordinates to the sample (Miller) coordinates.
     
    This function determines the rotation matrix from the dislocation coordinates to the sample coordinates (Miller indices). The sample coordinates always define the y||n and z||t, the x is defined based on y and z, since the Burger's vector could be in any direction.
    
    The input vectors don't have to be normalized before calling this function, but the orthogonality will be checked, and the function also makes sure both the Burger's vector and the line direction vector are both on the slip plane.

    Important Notes:
    * n and t must be given to determine the coordinate system
    * Otherwise, the function will return all the slip systems as a (3,3,12) array.

    Parameters
    ----------
    b : array of length 3, by default None
        Burger's vector in Miller, doesn't have to be normalized
    n : array of length 3, by default None
        normal vector of the slip plane, doesn't have to be normalized
    t : array of length 3, by default None
        line direction of the dislocation, doesn't have to be normalized

    Returns
    -------
    Ud : array of shape (3,3) or (3,3,12)
        rotation matrix that converts dislocation coordinates into the sample coordinates 
        if b, n, and t are given, Ud is (3,3)
        if b, n, and t are not given, all the slip systems are returned as (3,3,12)
    '''

    if (n is None) or (t is None):
        return return_dis_grain_matrices_all()
    if np.dot(n, t) != 0:
        raise ValueError('return_dis_grain_matrices: t must be on the plane n')
    yd = n/np.linalg.norm(n)
    zd = t/np.linalg.norm(t)
    xd = np.cross(yd, zd)
    Ud = np.transpose([xd, yd, zd])
    return Ud

class dispgrad_structure():
    ''' Displacement gradient function class.

    Attributes:
        d (dict): Input dictionary.
        b (float): Burger's vector magnitude.
        nu (float): Poisson's ratio.
    '''

    def __init__(self, d=default_dispgrad_dict()):
        self.d = d              # input dictionary

        # Burger's vector magnitude
        if 'b' in d:
            self.b = d['b']
        else:
            self.b = 1

        # Poisson's ratio
        if 'nu' in d.keys():
            self.nu = d['nu']
        else:
            self.nu = 0.334


class simple_shear(dispgrad_structure):
    ''' Wrapper for the simple shear deformation gradient function. '''
    def __init__(self, d=default_dispgrad_dict('simple_shear')):
        super().__init__(d)
        errorlist = []
        if 'R0' in d:
            self.R0 = d['R0']
        else:
            errorlist.append('R0')
        if 'components' in d:
            self.components = d['components']
        else:
            errorlist.append('components')
        if 'strain_magnitude' in d:
            self.strain_magnitude = d['strain_magnitude']
        else:
            errorlist.append('strain_magnitude')
        
        if len(errorlist) > 0:
            errorstr = ', '.join(errorlist)
            raise ValueError("simple_shear.__init__(): please define %s in d:dict"%errorstr)
        
    def Fg(self, xg, yg, zg):
        ''' Returns the displacement gradient tensor for the pure shear case.

        Parameters
        ----------
        xg : float, or array
            x coordinate in the grain system
        yg : float, or array (shape must match with xg)
            y coordinate in the grain system
        zg : float, or array (shape must match with xg)
            z coordinate in the grain system

        Returns
        -------
        Fd : numpy array
            xg.shape + (3, 3) strain tensor
        '''
        Fg = np.zeros(xg.shape + (3, 3))
        r = np.sqrt(xg**2 + yg**2 + zg**2)
        rind = r < self.R0
        i, j = self.components
        Fg[rind, i, j] = self.strain_magnitude
        Fg += np.eye(3) # add the identity tensor to convert to the deformation tensor

        return Fg

class edge_disl(dispgrad_structure):
    ''' Wrapper for the edge dislocation deformation gradient function. '''
    def __init__(self, d=default_dispgrad_dict('edge_disl')):
        super().__init__(d)

        errorlist = []
        # Burger's vector
        if 'bg' in d:
            bs = d['bg']
            self.bg = np.divide(bs, np.linalg.norm(bs))
        else:
            errorlist.append('bg')
        # normal vector of the slip plane
        if 'ng' in d:
            ns = d['ng']
            self.ng = np.divide(ns, np.linalg.norm(ns))
        else:
            errorlist.append('ng')
        # dislocation line direction
        if 'tg' in d:
            ts = d['tg']
            self.tg = np.divide(ts, np.linalg.norm(ts))
        else:
            errorlist.append('tg')

        if len(errorlist) > 0:
            errorstr = ', '.join(errorlist)
            raise ValueError("edge_disl.__init__(): please define %s in d:dict"%errorstr)

        # get the rotation matrix for the dislocation coordinates
        self.Ud = return_dis_grain_matrices(b=self.bg, n=self.ng, t=self.tg) # shape (3, 3)

        # Define grain rotation Ug (done in dispgrad_structure.__init__())

    def get_disl_strain_tensor(self, xd, yd):
        '''
        Returns dislocation strain tensor
        (currently just for edge-type dislocation)

        z: dislocation line direction
        y: normal of the slip plane
        x: Burger's vector direction (edge dislocation)

        Parameters
        ----------
        xd : float, or array
            x displacement from dislocation core
        yd : float, or array (shape must match with xd)
            y displacement from dislocation core

        Returns
        -------
        Fd : numpy array
            xd.shape + (3, 3) strain tensor
        '''
        prefactor = self.b/(4*np.pi*(1-self.nu))
        A = 2 * self.nu * (xd**2 + yd**2)       # xd.shape()
        denom = (xd**2 + yd**2)**2              # xd.shape()

        # All the following becomes xd.shape()x1x1
        Fxx = (-prefactor * yd * (3 * xd**2 + yd**2 - A) / denom)[..., None, None]
        Fxy = (prefactor * xd * (3 * xd**2 + yd**2 - A) / denom)[..., None, None]
        Fyx = (-prefactor * xd * (xd**2 + 3 * yd**2 - A) / denom)[..., None, None]
        Fyy = (prefactor * yd * (xd**2 - yd**2 - A) / denom)[..., None, None]

        O = np.zeros_like(Fxx)                  # zero-filler
        Fd = np.block([[Fxx, Fxy, O], [Fyx, Fyy, O], [O, O, O]]) + np.identity(3)
        return Fd

    def Fg(self, xg, yg, zg):
        ''' Returns the strain tensor of a single edge dislocation in the grain coordinates (Miller indices).
        
        The dislocation strain tensor is calculated in the dislocation coordinates, and then rotated into the grain coordinates.

        Parameters
        ----------
        xg : float, or array
            x displacement from dislocation core
        yg :float, or array (shape must match with xg)
            y displacement from dislocation core
        zg :float, or array (shape must match with xg)
            z displacement from dislocation core

        Returns
        -------
        Fg : numpy array
            shape(xg)x3x3 strain tensor
        '''

        rg = np.stack([xg, yg, zg], axis=-1) # shape (xg.shape, 3)
        rd = np.einsum('ij,...j->...i', self.Ud.T, rg) # shape (xg.shape, 3)
        xd, yd = rd[...,0], rd[...,1] # shape (xg.shape)
        # get strain tensor in the dislocation coordinates
        Fd = self.get_disl_strain_tensor(xd, yd)
        # rotate into the grain coordinates (Miller indices)
        Fg = np.einsum('ij,...jk,kl->...il', self.Ud, Fd, self.Ud.T)
        
        return Fg

class disl_network(dispgrad_structure):
    ''' Wrapper for the dislocation network deformation gradient function. '''
    def __init__(self, d=default_dispgrad_dict('disl_network')):
        super().__init__(d)
        self.a = d['a']         # non-singular radius
        if 'rn' in d:
            self.rn = d['rn']       # position of the nodes
        if 'links' in d:
            self.links = d['links'] # connectivity of the segments

    def load_network(self, filename, scale_cell=1.0, select_seg=None, verbose=False):
        ''' Load dislocation network from a ParaDiS restart file.

            rn: position of the nodes               (nNodes, 3)
            links: connectivity of the segments     (nLinks, 2)
            b: Burgers vector of the segments       (nLinks, 3)
            n: slip plane normal of the segments    (nLinks, 3)

        Parameters
        ----------
        filename : str
            filename of the ParaDis restart file
        scale_cell : float, optional
            scale the cell size. The default is 1.0.
        select_seg : list, optional
            select a subset of the segments. The default is None.
        verbose : bool, optional
            Print out the intermediate results for debugging.
        '''
        rn, links, cell = dio.read_vtk(filename, scale_cell=scale_cell, verbose=verbose, select_seg=select_seg)

        self.d['rn'] = self.rn = rn
        self.d['links'] = self.links = links
        self.d['cell'] = self.cell = cell

    def write_network_ca(self, filename, origin=None, bmag=None, reduced=False, pbc=False):
        """ Write Crystal Analysis file

        Parameters
        ----------
        filename : str
            filename of the Crystal Analysis file
        origin : tuple, optional
            origin of the simulation cell. The default is (0, 0, 0) at the center.
        bmag : float, optional
            magnitude of the Burger's vector. The default is None.
        reduced : bool, optional
            reduce discrete nodes on a single dislocation arm. The default is False.
        """
        if bmag is None:
            bmag = self.d['b']
        if origin is None:
            origin = tuple(-np.diag(self.cell)/2)
        ca_data = {}
        if reduced:
            ca_data['rn'], ca_data['links'], ca_data['cell'], ca_data['disl_list'], ca_data['disl_edge_list'] = dio.group_segments(filename, self.rn, self.links, self.cell, origin=origin, bmag=bmag, pbc=pbc)
        else:
            ca_data['rn'], ca_data['links'], ca_data['cell'] = dio.write_ca(filename, self.rn, self.links, self.cell, origin=origin, bmag=bmag)
        return ca_data

    def displacement_gradient_seg(self, b, r1, r2, r, verbose=False):
        ''' Calculate the displacement gradient tensor of a dislocation segment
        optimized by vectorization, now support multiple observation points

        The length unit is normalized to the Burger's vector magnitude

        Parameters
        ----------
        b : numpy array (3, )
            Burgers vector of the dislocation segment
        r1 : numpy array (3, )
            First endpoint of the dislocation segment
        r2 : numpy array (3, )
            Second endpoint of the dislocation segment
        r : numpy array (3, ) or (nobs, 3)
            Observation points
        verbose : bool, optional
            Print out the intermediate results for debugging.
            The default is False.

        Returns
        -------
        dudx : numpy array
            Displacement gradient tensor
        ''' 
        if len(r.shape) == 1:
            r = r.reshape(1, 3)
        r1 = r1.reshape(1, 3)
        r2 = r2.reshape(1, 3)
        nobs = r.shape[0]
        dudx = np.zeros((nobs, 3, 3))

        t = r2 - r1                 # (1, 3)
        t = t/np.linalg.norm(t)     # (1, 3)
        R = r1 - r                  # (n, 3)
        dr = np.dot(R, t.T)         # (n, 1)
        x0 = r1 - dr*t              # (n, 3)
        d = R - dr*t                # (n, 3)
        s1 = np.dot(r1 - x0, t.T)   # (n, 1)
        s2 = np.dot(r2 - x0, t.T)   # (n, 1)

        a2 = self.a**2              # (1, )
        d2 = np.sum(d*d, axis=1, keepdims=True)  # (n, 1)
        da2 = d2 + a2               # (n, 1)
        da2inv = 1/da2              # (n, 1)
        Ra1 = np.sqrt(s1*s1 + da2)  # (n, 1)
        Ra2 = np.sqrt(s2*s2 + da2)  # (n, 1)
        Ra1inv = 1/Ra1              # (n, 1)
        Ra1inv3 = Ra1inv**3         # (n, 1)
        Ra2inv = 1/Ra2              # (n, 1)
        Ra2inv3 = Ra2inv**3         # (n, 1)
        if verbose:
            print('r', r, 'Ra1', Ra1, 'Ra2', Ra2)

        J03 = da2inv*(s2*Ra2inv - s1*Ra1inv)                # (n, 1)
        J13 = -Ra2inv + Ra1inv                              # (n, 1)
        J15 = -1/3*(Ra2inv3 - Ra1inv3)                      # (n, 1)
        J25 = 1/3*da2inv*(s2**3*Ra2inv3 - s1**3*Ra1inv3)    # (n, 1)
        J05 = da2inv*(2*J25 + s2*Ra2inv3 - s1*Ra1inv3)      # (n, 1)
        J35 = 2*da2*J15 - s2**2*Ra2inv3 + s1**2*Ra1inv3     # (n, 1)
        if verbose:
            print('r', r, 'J03', J03, 'J13', J13, 'J15', J15, 'J25', J25, 'J05', J05, 'J35', J35)

        delta = np.eye(3)                                   # (3, 3)
        A = 3*a2*d*J05 + 2*d*J03 + 3*a2*t*J15 + 2*t*J13     # (n, 3)
        if verbose:
            print('r', r, 'A', A)

        B1 = (np.einsum('mj,kl->kjlm', delta, d) + np.einsum('jl,km->kjlm', delta, d) + np.einsum('lm,kj->kjlm', delta, d)) * J03.reshape(nobs, 1, 1, 1) # (n, 3, 3, 3) 
        B2 = (np.einsum('mj,kl->kjlm', delta, t) + np.einsum('jl,km->kjlm', delta, t) + np.einsum('lm,kj->kjlm', delta, t)) * J13.reshape(nobs, 1, 1, 1) # (n, 3, 3, 3)
        B3 = -3*np.einsum('km,kj,kl->kjlm', d, d, d) * J05.reshape(nobs, 1, 1, 1) # (n, 3, 3, 3)
        B4 = -3*(np.einsum('km,kj,kl->kjlm', d, d, t) + np.einsum('km,kj,kl->kjlm', d, t, d) + np.einsum('km,kj,kl->kjlm', t, d, d)) * J15.reshape(nobs, 1, 1, 1) # (n, 3, 3, 3)
        B5 = -3*(np.einsum('km,kj,kl->jlm', d, t, t) + np.einsum('km,kj,kl->kjlm', t, d, t) + np.einsum('km,kj,kl->kjlm', t, t, d)) * J25.reshape(nobs, 1, 1, 1) # (n, 3, 3, 3)
        B6 = -3*np.einsum('km,kj,kl->kjlm', t, t, t) * J35.reshape(nobs, 1, 1, 1) # (n, 3, 3, 3)
        B  = B1 + B2 + B3 + B4 + B5 + B6                    # (n, 3, 3, 3)

        Ab = np.einsum('mi,mj,k->mijk', A, t, b)            # (n, 3, 3, 3)
        
        U1 = np.zeros((nobs, 3, 3))
        U2 = np.zeros((nobs, 3, 3))
        U3 = np.zeros((nobs, 3, 3))
        for l in range(3):
            U1[:, l, 0] = Ab[:, 2, 1, l] - Ab[:, 1, 2, l]
            U1[:, l, 1] = Ab[:, 0, 2, l] - Ab[:, 2, 0, l]
            U1[:, l, 2] = Ab[:, 1, 0, l] - Ab[:, 0, 1, l]
            U2[:, 0, l] = Ab[:, l, 2, 1] - Ab[:, l, 1, 2]
            U2[:, 1, l] = Ab[:, l, 0, 2] - Ab[:, l, 2, 0]
            U2[:, 2, l] = Ab[:, l, 1, 0] - Ab[:, l, 0, 1]
            for m in range(3):
                U3[:, m, l] \
                    = (B[:, 1, l, m]*t[:, 2] - B[:, 2, l, m]*t[:, 1])*b[0] \
                    + (B[:, 2, l, m]*t[:, 0] - B[:, 0, l, m]*t[:, 2])*b[1] \
                    + (B[:, 0, l, m]*t[:, 1] - B[:, 1, l, m]*t[:, 0])*b[2]

        dudx = -1/8/np.pi*(U1 + U2 + 1/(1 - self.d['nu']) * U3)
        return dudx

    def displacement_gradient_structure(self, r, verbose=False, zeros=False):
        '''Computes the non-singular displacement gradient produced by the dislocation structure.

        The length unit is normalized to the Burger's vector magnitude

        Parameters
        ----------
        r : numpy array (nobs, 3)
            Observation point
        verbose : bool, optional
            Print out the intermediate results for debugging.
        zeros : bool, optional
            If True, the function will return a zero-filled array for testing purpose.

        Returns
        -------
        dudx : numpy array (nobs, 3, 3)
            Displacement gradient tensor
        '''
        nobs = r.shape[0]
        dudx = np.zeros((nobs, 3, 3))
        if not zeros:
            for i in range(self.links.shape[0]):
                n1 = self.links[i, 0].astype(int)
                n2 = self.links[i, 1].astype(int)
                r1 = self.rn[n1, :]
                r2 = self.rn[n2, :]
                b = self.links[i, 2:5]
                dudx += self.displacement_gradient_seg(b, r1, r2, r, verbose=verbose)
        
        return dudx

    def Fg(self, xg, yg, zg, filename=None, zeros=False, verbose=False):
        ''' Returns the dislocation strain tensor in the grain (Miller) coordinates.

        Use the non-singular displacement gradient function

        Parameters
        ----------
        xg : float, or array
            x coordinate in the grain system (m)
        yg :float, or array (shape must match with xg)
            y coordinate in the grain system (m)
        zg :float, or array (shape must match with xg)
            z coordinate in the grain system (m)
        filename : str, optional
            filename to load the displacement gradient field from file. The default is None.
        zeros : bool, optional
            If True, the function will return a zero-filled array for testing purpose.
        verbose : bool, optional
            Print out the intermediate results for debugging.

        Returns
        -------
        Fg : numpy array
            shape(xg)x3x3 strain tensor
        '''
        
        # Set up parameters
        bmag = self.d['b']

        # Normalize the coordinate into the unit of the Burgers vector
        r_obs = np.stack([xg.flatten(), yg.flatten(), zg.flatten()], axis=-1)
        rnorm = r_obs/bmag
        # self.d['rn'] = self.rn = self.rn/bmag

        # Load the displacement gradient field from file
        if filename is not None and os.path.exists(filename):
            if verbose:
                print('Loading displacement gradient from file %s' % filename)
            Fg_list = np.load(filename)['Fg']
        else:
            if verbose:
                print('Calculating displacement gradient')
            # Fg_list = self.displacement_gradient_structure(rnorm, zeros=zeros, verbose=verbose)
            Fg_list = np.zeros((rnorm.shape[0], 3, 3))
            if not zeros:
                Fg_list = dgh.displacement_gradient_structure_matlab(self.rn, self.links, self.d['nu'], self.a, rnorm)
                # Fg_list = dgh.displacement_gradient_structure(self.rn, self.links, self.d['nu'], self.a, rnorm)
            if filename is not None:
                np.savez_compressed(filename, Fg=Fg_list, r_obs=r_obs)

        Fg = np.reshape(Fg_list, xg.shape + (3, 3)) + np.identity(3)
        return Fg
