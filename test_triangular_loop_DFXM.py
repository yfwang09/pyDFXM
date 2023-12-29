#%%-----------------------------------------------------------------
# DFXM forward calculation for a triangular dislocation loop
#
# The test case for the displacement gradient field is in test_triangular_loop_dispgrad.py
#
# Note that rn defined in the disl object is in the unit of Burger's vector,
# while the r in the DFXM calculation is in the unit of meter.
# The bmag is only used in the Fg function for DFXM calculation 
# (in test_triangular_loop_DFXM.py)
#
# Developer: Yifan Wang, yfwang09@stanford.edu
# Date: 2023/12/27
#---------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
import dispgrad_func as dgf
import forward_model as fwd
import visualize_helper as vis

#%%-------------------------------------------------------
# INITIALIZATION
#---------------------------------------------------------

# Elasticity parameters (Aluminum)
input_dict = dgf.default_dispgrad_dict('disl_network')

input_dict['nu'] = NU = 0.324       # Poisson's ratio
input_dict['b'] = bmag = 2.86e-10   # Burger's magnitude (m)

# Initialize the triangular loop 
L = 20000        # in the unit of b
# rn = L*(np.random.rand(3, 3) - 0.5)
rn = np.array([[ 78.12212123, 884.74707189, 483.30385117],
               [902.71333272, 568.95913492, 938.59105117],
               [500.52731411, 261.22281654, 552.66098404]])*20 - L/2
print(rn)
# Normalized Burger's vector
b = np.array([1, 1, 0])
b = b/np.linalg.norm(b)
# Normalized slip plane normal
n = np.array([1, 1, 1])
n = n/np.linalg.norm(n)
# Connectivity
links = np.transpose([[0, 1, 2], [1, 2, 0]])
links = np.hstack([links, np.tile(b, (3, 1)), np.tile(n, (3, 1))])

# Initialize the dislocation network object
input_dict['rn'] = rn
input_dict['links'] = links
disl = dgf.disl_network(input_dict)

#---------------------------------------------------------
# Plot the dislocation loop
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# lb, ub = -L/2*bmag*1e6, L/2*bmag*1e6
# ax.plot([lb, lb, ub, ub], [ub, ub, ub, ub], [lb, ub, ub, lb], 'k')
# ax.plot([lb, lb, ub, ub], [lb, ub, ub, lb], [lb, lb, lb, lb], 'k')
# for i in range(links.shape[0]):
#     n12 = links[i, 0:2].astype(int)
#     r12 = rn[n12, :]*bmag*1e6
#     ax.plot(r12[..., 0], r12[..., 1], r12[..., 2],  'C3o-')
# ax.plot([lb, lb, ub, ub], [lb, lb, lb, lb], [ub, lb, lb, ub], 'k')
# ax.plot([lb, lb, ub, ub], [ub, lb, lb, ub], [ub, ub, ub, ub], 'k')
# plt.show()

#%%-------------------------------------------------------
# CALCULATE THE DISPLACEMENT GRADIENT
#---------------------------------------------------------

forward_dict = fwd.default_forward_dict()
print(forward_dict)

# Set up the pre-calculated resolution function
datapath = 'data'
os.makedirs(datapath, exist_ok=True)
saved_res_fn = os.path.join(datapath, 'Res_qi_Al_001.npz')
print('saved resolution function at %s'%saved_res_fn)

model = fwd.DFXM_forward(forward_dict, load_res_fn=saved_res_fn)
Ug = model.Ug
print('Ug')
print(Ug)

lb, ub = -L/2*bmag, L/2*bmag            # in unit of m
Ngrid = 50
xs = np.linspace(lb, ub, Ngrid)         # (Ngrid, )
ys = np.linspace(lb, ub, Ngrid)         # (Ngrid, )
zs = np.linspace(lb, ub, Ngrid)         # (Ngrid, )

XX, YY, ZZ = np.meshgrid(xs, ys, zs)    # (Ngrid, Ngrid, Ngrid)
Rs = np.stack([XX,YY,ZZ], -1)           # (Ngrid, Ngrid, Ngird, 3)
print('Grid size in the sample coordinates:', Rs.shape)

print('Convert Rs into the grain coordinates (Miller indices)')
Rg = np.einsum('ij,...j->...i', Ug, Rs)
Fg = disl.Fg(Rg[..., 0], Rg[..., 1], Rg[..., 2])

print('Visualize the displacement gradient')
i, j = 1, 2
vmin, vmax = -1e-3, 1e-3
fs = 12
extent = np.multiply(1e6, [lb, ub, lb, ub, lb, ub]) # in the unit of um
figax = vis.plot_3d_slice_z(Fg[:, :, :, i, j], extent=extent, vmin=vmin, vmax=vmax, nslice=5, fs=fs, show=False)
fig, ax = figax
for i in range(links.shape[0]):
    n12 = links[i, 0:2].astype(int)
    r12 = rn[n12, :]*bmag*1e6 # in unit of um
    ax.plot(r12[..., 0], r12[..., 1], r12[..., 2],  'ko-')
# change the view angle
# ax.view_init(elev=90, azim=-90)
ax.set_title(r'Grain coordinate system, $F^g_{yz}$', fontsize=fs)
ax.set_xlabel(r'${\rm x(\mu m)}$')
ax.set_ylabel(r'${\rm y(\mu m)}$')
ax.set_zlabel(r'${\rm z(\mu m)}$')
plt.show()

#%%-------------------------------------------------------
# CALCULATE THE DISPLACEMENT GRADIENT
#---------------------------------------------------------

print('#'*20 + ' Calculate and visualize the image')
saved_Fg_file = os.path.join(datapath, 'Fg_triloop_Al_001.npz')
print('saved displacement gradient at %s'%saved_Fg_file)
Fg_func = lambda x, y, z: disl.Fg(x, y, z, filename=saved_Fg_file)
im, ql, rulers = model.forward(Fg_func)

# Visualize the simulated image
figax = vis.visualize_im_qi(forward_dict, im, None, rulers) #, vlim_im=[0, 200])

# Visualize the reciprocal space wave vector ql
# figax = vis.visualize_im_qi(forward_dict, None, ql, rulers, vlim_qi=[-1e-4, 1e-4])

# Visualize the observation points
uc = 1e6 # um/m
r_obs = np.load(saved_Fg_file)['r_obs'] * uc
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
lb, ub = -L/2*bmag * uc, L/2*bmag * uc
ax.plot([lb, lb, ub, ub], [ub, ub, ub, ub], [lb, ub, ub, lb], 'k')
ax.plot([lb, lb, ub, ub], [lb, ub, ub, lb], [lb, lb, lb, lb], 'k')
for i in range(links.shape[0]):
    n12 = links[i, 0:2].astype(int)
    r12 = rn[n12, :]*bmag * uc
    ax.plot(r12[..., 0], r12[..., 1], r12[..., 2],  'C3o-')

nskip = 10
ax.plot(r_obs[::nskip, 0], r_obs[::nskip, 1], r_obs[::nskip, 2],  'C0.', markersize=0.01)
ax.plot([lb, lb, ub, ub], [lb, lb, lb, lb], [ub, lb, lb, ub], 'k')
ax.plot([lb, lb, ub, ub], [ub, lb, lb, ub], [ub, ub, ub, ub], 'k')
ax.axis('equal')
plt.show()
