#%%-----------------------------------------------------------------
# DFXM forward calculation for a single dislocation in diamond
#
# Developer: Yifan Wang, yfwang09@stanford.edu
# Date: 2024/02/12
#---------------------------------------------------------------

import os, time
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

forward_dict = fwd.default_forward_dict()
forward_dict['two_theta'] = two_theta
print(forward_dict)

# Set up the pre-calculated resolution function
datapath = 'data'
os.makedirs(datapath, exist_ok=True)
saved_res_fn = os.path.join(datapath, 'Res_qi_diamond_001.npz')
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
saved_Fg_file = os.path.join(datapath, 'Fg_%s_DFXM_robs.npz'%casename)
print('saved observation points at %s'%saved_Fg_file)
Fg_func = lambda x, y, z: disl.Fg(x, y, z, filename=saved_Fg_file)
if not os.path.exists(saved_Fg_file):
    im, ql, rulers = model.forward(Fg_func, timeit=True)

# Visualize the observation points
Lx, Ly, Lz = tuple(np.diag(disl.cell))
lbx, ubx = -Lx/2*bmag, Lx/2*bmag         # in unit of m
lby, uby = -Ly/2*bmag, Ly/2*bmag         # in unit of m
lbz, ubz = -Lz/2*bmag, Lz/2*bmag         # in unit of m
extent = np.multiply(1e6, [lbx, ubx, lby, uby, lbz, ubz]) # in the unit of um
fig, ax = vis.visualize_disl_network(disl.d, disl.rn, disl.links, extent=extent, unit='um', show=False)
nskip = 10
r_obs = np.load(saved_Fg_file)['r_obs']*1e6 # in the unit of um
ax.plot(r_obs[::nskip, 0], r_obs[::nskip, 1], r_obs[::nskip, 2],  'C0.', markersize=0.01)
plt.show()

# %% --------------------------------------------------------
# CALCULATE THE DFXM IMAGE
# -----------------------------------------------------------

bvec = [1, 1, 0]
nvec = [1,-1, 1]
rt = [0, 30, 60, 90] # rotate angle of xi
rvec = [0, 0, 1]
for th in rt:
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

    print('#'*10 + ' Calculate and visualize the image for xi = %s'%xi)

    saved_Fg_file = os.path.join(datapath, 'Fg_%s_rvec%d%d%d_%d_DFXM.npz'%((casename, ) + tuple(rvec) + (th,)))
    print('saved displacement gradient at %s'%saved_Fg_file)
    Fg_func = lambda x, y, z: disl.Fg(x, y, z, filename=saved_Fg_file)
    im, ql, rulers = model.forward(Fg_func, timeit=True)

    # Visualize the simulated image
    figax = vis.visualize_im_qi(forward_dict, im, None, rulers) #, vlim_im=[0, 200])

# Visualize the reciprocal space wave vector ql
# figax = vis.visualize_im_qi(forward_dict, None, ql, rulers, vlim_qi=[-1e-4, 1e-4])

# Visualize the observation points
# extent = np.multiply(1e6, [lbx, ubx, lby, uby, lbz, ubz]) # in the unit of um
# fig, ax = vis.visualize_disl_network(disl.d, disl.rn, disl.links, extent=extent, unit='um', show=False)
# nskip = 10
# r_obs = np.load(saved_Fg_file)['r_obs']*1e6 # in the unit of um
# ax.plot(r_obs[::nskip, 0], r_obs[::nskip, 1], r_obs[::nskip, 2],  'C0.', markersize=0.01)
# plt.show()

# 

# r_obs_xyz_file = os.path.join(datapath, 'r_obs_%s.xyz'%casename)
# im_Nobs = np.repeat(np.repeat(im, Nobs, axis=0), Nobs, axis=1)[:,:,np.newaxis]
# im_obs = np.tile(im_Nobs, (1, 1, model.d['Npixels'][2]*Nobs)).reshape((-1, 1))
# r_obs = r_obs_cell.reshape((-1, 3))
# dio.write_xyz(r_obs_xyz_file, r_obs, props=im_obs, scale=1e10)

