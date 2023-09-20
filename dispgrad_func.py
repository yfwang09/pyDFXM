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
# import displacement_grad_helper as dgh

def default_dispgrad_dict(dispgrad_type='simple_shear'):
    '''Generate a default displacement gradient dictionary'''
    dispgrad_dict = {
    # Materials properties
    'b': 1, 'nu': 0.334,
    # Grain rotation (Ug, Eq. 7-8)
    'x_c': [1,0,0],     # x dir. for the crystal system (Fig.2)
    'y_c': [0,1,0],     # y dir. for the crystal system (Fig.2)
    'hkl': [0,0,1],     # hkl diffraction plane, z dir. crystal
    ## 'Ug': np.identity(3), # or directly define a rotation matrix
    # Displacement gradient (strain field) type
    'dispgrad_type': dispgrad_type,
    ## 'dispgrad_type': 'simple_shear', # simple shear in a sphere
    ## 'dispgrad_type': 'edge_disl', # single edge dislocation case
    ## 'dispgrad_type': 'disl_network', # dislocation network (todo)
    }
    if dispgrad_dict['dispgrad_type'] == 'simple_shear':
        dispgrad_dict['R0'] = dispgrad_dict['b']*10000
        dispgrad_dict['components'] = (2, 0)
        dispgrad_dict['strain_magnitude'] = 5e-4
    elif dispgrad_dict['dispgrad_type'] == 'edge_disl':
        dispgrad_dict['bs'] = [1,-1, 0] # Burger's vector dir. in Miller (sample)
        dispgrad_dict['ns'] = [1, 1,-1] # Normal vector dir. in Miller (sample)
        dispgrad_dict['ts'] = [1, 1, 2] # Dislocation line dir. in Miller (sample)
    elif dispgrad_dict['dispgrad_type'] == 'disl_network':
        # print('dispgrad_type == disl_network is not implemented')
        pass
    else:
        raise NotImplementedError('dispgrad_type == %s is not implemented'%dispgrad_dict['dispgrad_type'])

    return dispgrad_dict

def return_dis_grain_matrices_all():
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
    
    The vectors don't have to be normalized, but the orthogonality will be checked, and the function also makes sure both the Burger's vector and the line direction vector are both on the slip plane.

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
    ''' Displacement gradient function class. '''
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

        # get the grain rotation matrix
        if 'Ug' in d.keys():
            self.Ug = d['Ug']
        elif 'hkl' in d.keys():
            z_c = d['hkl']
            if 'x_c' in d.keys():
                x_c = d['x_c']
            else:
                x_c = [1, 0, 0]
            if 'y_c' in d.keys():
                y_c = d['y_c']
            else:
                y_c = np.cross(z_c, x_c)
                x_c = np.cross(y_c, z_c)
            self.Ug = return_dis_grain_matrices(b=x_c, n=y_c, t=z_c).T
        else:
            self.Ug = np.eye(3)

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
        if 'bs' in d:
            bs = d['bs']
            self.bs = np.divide(bs, np.linalg.norm(bs))
        else:
            errorlist.append('bs')
        # normal vector of the slip plane
        if 'ns' in d:
            ns = d['ns']
            self.ns = np.divide(ns, np.linalg.norm(ns))
        else:
            errorlist.append('ns')
        # dislocation line direction
        if 'ts' in d:
            ts = d['ts']
            self.ts = np.divide(ts, np.linalg.norm(ts))
        else:
            errorlist.append('ts')

        if len(errorlist) > 0:
            errorstr = ', '.join(errorlist)
            raise ValueError("edge_disl.__init__(): please define %s in d:dict"%errorstr)

        # get the rotation matrix for the dislocation coordinates
        self.Ud = return_dis_grain_matrices(b=self.bs, n=self.ns, t=self.ts) # shape (3, 3)

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
        ''' Returns the strain tensor of a single edge dislocation in the sample coordinates.
        
        The dislocation strain tensor is calculated in the dislocation coordinates, and then rotated into the sample coordinates.

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
        Fd : numpy array
            shape(xg)x3x3 strain tensor
        '''

        rg = np.stack([xg, yg, zg], axis=-1) # shape (xg.shape, 3)
        rc = np.einsum('ij,...j->...i', self.Ug, rg) # shape (xg.shape, 3)
        rd = np.einsum('ij,...j->...i', self.Ud.T, rc) # shape (xg.shape, 3)
        xd, yd = rd[...,0], rd[...,1] # shape (xg.shape)
        # get strain tensor in the dislocation coordinates
        Fd = self.get_disl_strain_tensor(xd, yd)
        # rotate into the crystal coordinates (Miller indices)
        Fc = np.einsum('ij,...jk,kl->...il', self.Ud, Fd, self.Ud.T)
        # rotate into the grain coordinates
        Fg = np.einsum('ij,...jk,kl->...il', self.Ug, Fc, self.Ug.T)
        
        return Fg

class disl_network(dispgrad_structure):
    ''' Wrapper for the dislocation network deformation gradient function. '''
    def __init__(self, d=default_dispgrad_dict('disl_network')):
        super().__init__(d)


def Fg_disl_network(d, xg, yg, zg, filename=None):
    ''' Returns the dislocation strain tensor in the sample coordinates.

    Use the non-singular displacement gradient function

    Parameters
    ----------
    d : dict
        resolution function input dictionary
    xg : float, or array
        x coordinate in the grain system
    yg :float, or array (shape must match with xg)
        y coordinate in the grain system
    zg :float, or array (shape must match with xg)
        z coordinate in the grain system

    Returns
    -------
    Fg : numpy array
        shape(xg)x3x3 strain tensor
    '''
    # get grain rotation matrix
    if 'Ug' in d.keys():
        Ug = d['Ug']
    else:
        Ug = np.eye(3)
    if ('hkl' in d.keys()) and ('xcry' in d.keys()) and ('ycry' in d.keys()):
        Ug = return_dis_grain_matrices(b=d['xcry'], n=d['ycry'], t=d['hkl']).T
    if 'nu' in d.keys():
        NU = d['nu']
    else:
        NU = 0.324
    if 'a' in d.keys():
        a = d['a']
    else:
        a = 1.0
    if 'rn' in d.keys():
        rn = d['rn']
    else:
        rn = np.array([[ 78.12212123, 884.74707189, 483.30385117],
                       [902.71333272, 568.95913492, 938.59105117],
                       [500.52731411, 261.22281654, 552.66098404]])
    if 'links' in d.keys():
        links = d['links']
    else:
        links = np.transpose([[0, 1, 2], [1, 2, 0]])
    if 'bs' in d.keys():
        b = d['bs']
        b = b/np.linalg.norm(b)
        # if 'b' in d.keys():
        #     b = b*d['b']
    else:
        b = np.array([1, 1, 0])
        b = b/np.linalg.norm(b)
    if 'b' in d.keys():
        bmag = d['b']
    else:
        bmag = 1
    if 'ns' in d.keys():
        n = d['ns']
    else:
        n = np.array([1, 1, 1])
    n = n/np.linalg.norm(n)

    if links.shape[1] == 2: # only connectivity is provided
        links = np.hstack([links, np.tile(b, (3, 1)), np.tile(n, (3, 1))])
    elif links.shape[1] != 8:
        raise ValueError('links array must include b and n')
    r_obs = np.stack([xg.flatten(), yg.flatten(), zg.flatten()], axis=-1)
    rnorm = r_obs/bmag
    rn = rn/bmag

    if filename is not None and os.path.exists(filename):
        print('Loading displacement gradient from file %s' % filename)
        Fg_list = np.load(filename)['Fg']
    else:
        test = (filename == 'test')
        Fg_list = dgh.displacement_gradient_structure_matlab(rn, links, NU, a, rnorm, test=test)
        np.savez_compressed(filename, Fg=Fg_list, r_obs=r_obs)

    # if filename is not None:
    #     np.savez_compressed(filename, Fg=Fg_list, r_obs=r_obs)
    Fg = np.reshape(Fg_list, xg.shape + (3, 3)) + np.identity(3)
    
    return Fg
