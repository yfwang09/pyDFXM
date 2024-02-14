#%%-----------------------------------------------------------------
# DFXM forward calculation for a single dislocation in diamond
#
# Developer: Yifan Wang, yfwang09@stanford.edu
# Date: 2024/02/12
#---------------------------------------------------------------

import os, time
from typing import Never
import numpy as np
import matplotlib.pyplot as plt
import dispgrad_func as dgf
import forward_model as fwd
import visualize_helper as vis
import disl_io_helper as dio

#%%-------------------------------------------------------
# INITIALIZATION
#---------------------------------------------------------

# Configuration files
casename = 'diamond_disl'

# Elasticity parameters (Diamond)
input_dict = dgf.default_dispgrad_dict('disl_network')
print(input_dict)

input_dict['nu'] = NU = 0.200       # Poisson's ratio
input_dict['b'] = bmag = 2.522e-10  # Burger's magnitude (m)
two_theta = 48.16                   # 2theta for diamond-(004) (deg)

# Load the dislocation network
disl = dgf.disl_network(input_dict)

#%%-------------------------------------------------------
# SETUP THE FORWARD MODEL
#---------------------------------------------------------

# bvec = [1, 0, 1]; nvec = [1, 1, -1]; rvec = [1, 1, -1]
# bvec = [1, 0, 1]; nvec = [1, 1, -1]; rvec = [1,-1,-1]

bvec = [1, 1, 0]; nvec = [1,-1, 1]; rvec = [0, 0, 1]
# bvec = [1, 1, 0]; nvec = [1,-1, 1]; rvec = [1,-1, 1]

# rt = np.arange(-180, 180, 30, dtype=int) # rotate angle of xi
rt = np.arange(0, 180, 30, dtype=int) # rotate angle of xi

bvecname = 'b%d%d%d'%(bvec[0], bvec[1], bvec[2])
nvecname = 'n%d%d%d'%(nvec[0], nvec[1], nvec[2])
rvecname = 'rvec%d%d%d'%(rvec[0], rvec[1], rvec[2])

forward_dict = fwd.default_forward_dict()
forward_dict['two_theta'] = two_theta
forward_dict['npoint1'] = forward_dict['npoint2'] = forward_dict['npoint3'] = 800
print(forward_dict)

# Set up the pre-calculated resolution function
datapath = os.path.join('data', casename + '_' + bvecname + '_' + nvecname)
os.makedirs(datapath, exist_ok=True)
saved_res_fn = os.path.join('data', 'Res_qi_diamond_001_1e-5.npz')
print('saved resolution function at %s'%saved_res_fn)

# Set up the pre-calculated displacement gradient
model = fwd.DFXM_forward(forward_dict, load_res_fn=saved_res_fn)
Ug = model.Ug
print('Ug')
print(Ug)

#%%-------------------------------------------------------
# CALCULATE THE OBSERVATION POINTS OF THE DFXM
#---------------------------------------------------------

L = 40000
disl.d['cell'] = disl.cell = np.diag([L, L, L])
disl.d['rn'] = disl.rn = np.array([])
disl.d['links'] = disl.links = np.array([])

# calculate the observation points
# saved_Fg_file = os.path.join(datapath, 'Fg_%s_DFXM_robs.npz'%casename)
# print('saved observation points at %s'%saved_Fg_file)
# Fg_func = lambda x, y, z: disl.Fg(x, y, z, filename=saved_Fg_file)
# if not os.path.exists(saved_Fg_file):
#     im, ql, rulers = model.forward(Fg_func, timeit=True)

# # Visualize the observation points
# Lx, Ly, Lz = tuple(np.diag(disl.cell))
# lbx, ubx = -Lx/2*bmag, Lx/2*bmag         # in unit of m
# lby, uby = -Ly/2*bmag, Ly/2*bmag         # in unit of m
# lbz, ubz = -Lz/2*bmag, Lz/2*bmag         # in unit of m
# extent = np.multiply(1e6, [lbx, ubx, lby, uby, lbz, ubz]) # in the unit of um
# fig, ax = vis.visualize_disl_network(disl.d, disl.rn, disl.links, extent=extent, unit='um', show=False)
# nskip = 10
# r_obs = np.load(saved_Fg_file)['r_obs']*1e6 # in the unit of um
# ax.plot(r_obs[::nskip, 0], r_obs[::nskip, 1], r_obs[::nskip, 2],  'C0.', markersize=0.01)
# plt.show()

# %% --------------------------------------------------------
# CALCULATE THE DFXM IMAGE
# -----------------------------------------------------------

from itertools import product

# phi_values = np.linspace(-0.001, 0.001, 21).round(4)
phi_values = np.array([-0.0002, -0.00015, -0.0001, -0.00005, 0, 0.00005, 0.0001, 0.00015, 0.0002])
phi_values = np.linspace(-0.0001, 0.0001, 11).round(5)
chi_values = [0, ]
# chi_values = np.linspace(-0.001, 0.001, 21).round(4)

savename = rvecname+'_'+bvecname+'_fine_phi_chi'

if os.path.exists(savename + '.npz'):
    rawdata = np.load(savename + '.npz')
    Imax, Imin = rawdata['Imax'], rawdata['Imin']
    rt = rawdata['rt']
else:
    Imax = np.zeros((len(rt), len(phi_values), len(chi_values)))
    Imin = np.zeros((len(rt), len(phi_values), len(chi_values)))

    # for th, phi, chi in product(rt, phi_values, chi_values):
    for i, j, k in product(range(len(rt)), range(len(phi_values)), range(len(chi_values))):
        th, phi, chi = rt[i], phi_values[j], chi_values[k]
        model.d['phi'] = phi
        model.d['chi'] = chi
        # rotate along rvec (Rodrigues' rotation formula)
        # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        kvec = np.divide(rvec, np.linalg.norm(rvec))
        K = np.array([[0, -kvec[2], kvec[1]], [kvec[2], 0, -kvec[0]], [-kvec[1], kvec[0], 0]])
        R = np.eye(3) + np.sin(np.deg2rad(th))*K + (1-np.cos(np.deg2rad(th)))*np.dot(K, K)
        xi = np.dot(R, bvec)

        rn, links, cell = dio.create_single_disl(xi, b=bvec, n=nvec, L=L, shift=[0,0,0])
        disl.d['rn'] = disl.rn = rn
        disl.d['links'] = disl.links = links
        disl.d['cell'] = disl.cell = cell

        print('#'*10 + ' Calculate and visualize the image for xi = %s, phi = %s, chi = %s'%(xi, phi, chi) + '#'*10)

        saved_Fg_file = os.path.join(datapath, 'Fg_%s_%s_%d_%.4f_%.4f_DFXM.npz'%(casename, rvecname, th, phi, chi))
        saved_Fg_file = os.path.join(datapath, 'Fg_%s_%s_%d_%.5f_%.5f_DFXM.npz'%(casename, rvecname, th, phi, chi))
        print('saved displacement gradient at %s'%saved_Fg_file)
        Fg_func = lambda x, y, z: disl.Fg(x, y, z, filename=saved_Fg_file)
        im, ql, rulers = model.forward(Fg_func, timeit=True)

        Imax[i, j, k], Imin[i, j, k] = np.max(im), np.min(im)

        # Visualize the simulated image
        figax = vis.visualize_im_qi(forward_dict, im, None, rulers, show=False) #, vlim_im=[0, 200])
        saved_im_file = os.path.join(datapath, 'im_%s_%s_%d_%.4f_%.4f_DFXM.png'%(casename, rvecname, th, phi, chi))
        saved_im_file = os.path.join(datapath, 'im_%s_%s_%d_%.5f_%.5f_DFXM.png'%(casename, rvecname, th, phi, chi))
        figax[0].savefig(saved_im_file, dpi=300)
        print('saved the image at %s'%saved_im_file)
        plt.close()

        # Visualize the observation points
        saved_obs_file = os.path.join(datapath, 'obs_%s_%s_%d_DFXM.png'%(casename, rvecname, th))
        if not os.path.exists(saved_obs_file):
            Lx, Ly, Lz = tuple(np.diag(disl.cell))
            lbx, ubx = -Lx/2*bmag, Lx/2*bmag         # in unit of m
            lby, uby = -Ly/2*bmag, Ly/2*bmag         # in unit of m
            lbz, ubz = -Lz/2*bmag, Lz/2*bmag         # in unit of m
            extent = np.multiply(1e6, [lbx, ubx, lby, uby, lbz, ubz]) # in the unit of um
            fig, ax = vis.visualize_disl_network(disl.d, disl.rn, disl.links, extent=extent, unit='um', show=False)
            nskip = 10
            r_obs = np.load(saved_Fg_file)['r_obs']*1e6 # in the unit of um
            ax.plot(r_obs[::nskip, 0], r_obs[::nskip, 1], r_obs[::nskip, 2],  'C0.', markersize=0.01)
            
            fig.savefig(saved_obs_file, dpi=300)
            print('saved the observation points at %s'%saved_obs_file)
            plt.close()

    np.savez_compressed(savename, Imax=Imax, Imin=Imin, phi_values=phi_values, chi_values=chi_values, rt=rt)

#%%
# print(Imax.shape)
fig, ax = plt.subplots()
# ax.plot(phi_values, Imax[:, :, 0].T, 'o-', label='Imax')
# ax.plot(phi_values, Imin[:, :, 0].T, 'o-', label='Imin')
for i in range(len(rt)):
    ax.plot(phi_values*1000, Imax[i, :, 0].T - Imin[i, :, 0].T, 'o-', label=r'$I_{\max} - I_{\min}(%d\degree)$'%rt[i])
    # ax.plot(phi_values, Imax[i, :, 0].T, '^-C%d'%i, label=r'$I_{\rm max}(%d\degree)$'%rt[i])
    # ax.plot(phi_values, Imin[i, :, 0].T, 'v-C%d'%i, label=r'$I_{\rm min}(%d\degree)$'%rt[i])
ax.set_xticks(phi_values[::2]*1000)
ax.set_ylabel('Intensity')
ax.set_xlabel(r'$\phi(\times0.001\degree)$')
ax.legend()

fig, ax = plt.subplots()
for iphi in range(phi_values.size//2, phi_values.size, 2):
    xval = rt
    yval = Imax[:, iphi, 0] - Imin[:, iphi, 0]
    ax.plot(np.append(xval, 180), np.append(yval, yval[0]), 'o-', label=r'$\phi=%.5f$'%phi_values[iphi])
    ax.set_ylabel('Intensity')
    ax.set_xlabel(r'$\alpha(\degree)$')
    ax.legend()
for ic, iphi in enumerate(range(phi_values.size//2, -1, -2)):
    xval = rt
    yval = Imax[:, iphi, 0] - Imin[:, iphi, 0]
    ax.plot(-np.append(xval, 180), np.append(yval, yval[0]), 'o-C%d'%ic, label=r'$\phi=%.5f$'%phi_values[iphi])
    ax.set_ylabel('Intensity')
    ax.set_xlabel(r'$\alpha(\degree)$')
    ax.legend()

# ax.set_xlim(0, 180)
# ax.set_xlim(-90, 90)
# ax.set_xlim(np.min(rt), np.max(rt) + 30)

plt.show()

# %%
# Compile the dislocations into an OVITO file

rn_full = []
links_full = []

for i, th in enumerate(rt):
    # rotate along rvec (Rodrigues' rotation formula)
    # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    kvec = np.divide(rvec, np.linalg.norm(rvec))
    K = np.array([[0, -kvec[2], kvec[1]], [kvec[2], 0, -kvec[0]], [-kvec[1], kvec[0], 0]])
    R = np.eye(3) + np.sin(np.deg2rad(th))*K + (1-np.cos(np.deg2rad(th)))*np.dot(K, K)
    xi = np.dot(R, bvec)

    rn, links, cell = dio.create_single_disl(xi, b=bvec, n=nvec, L=L, shift=[0,0,0])

    links[0, :2] += i*2
    rn_full.append(rn)
    links_full.append(links)

disl.d['rn'] = disl.rn = np.vstack(rn_full)
disl.d['links'] = disl.links = np.vstack(links_full)
disl.d['cell'] = disl.cell = cell
# print(disl.rn)
# print(disl.links)

config_ca_file = savename + '.ca'
if not os.path.exists(config_ca_file):
    ca_data = disl.write_network_ca(config_ca_file, bmag=bmag)

r_obs_xyz_file = 'robs.xyz'
saved_Fg_file = os.path.join('data', 'Fg_%s_DFXM_robs.npz'%casename)
if not os.path.exists(r_obs_xyz_file):
    r_obs = np.load(saved_Fg_file)['r_obs']
    dio.write_xyz(r_obs_xyz_file, r_obs, scale=1e10)
