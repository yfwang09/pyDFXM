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

#%%-------------------------------------------------------
# CALCULATE THE DISPLACEMENT GRADIENT
#---------------------------------------------------------

forward_dict = fwd.default_forward_dict()

# if you want to change the forward model setup
# forward_dict['hkl'] = [1, 1, 1]
# forward_dict['x_c'] = [1, -1, 0]
# forward_dict['y_c'] = [-1, 1, 0]
# forward_dict['phi'] = np.deg2rad(0.001)

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

# or you can change the parameters here
# model.d['phi'] = np.deg2rad(0.001)

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
figax = vis.visualize_disl_network(disl.d, rn, links, unit='um', figax=figax, show=False)
plt.show()

#%%-------------------------------------------------------
# CALCULATE THE DFXM IMAGE
#---------------------------------------------------------

print('#'*20 + ' Calculate and visualize the image')
saved_Fg_file = os.path.join(datapath, 'Fg_triloop_Al_001.npz')
print('saved displacement gradient at %s'%saved_Fg_file)

# Strong beam condition
Fg_func = lambda x, y, z: disl.Fg(x, y, z, filename=saved_Fg_file)
im, ql, rulers = model.forward(Fg_func)

# Visualize the simulated image
figax = vis.visualize_im_qi(forward_dict, im, None, rulers) #, vlim_im=[0, 200])

# Rocking curve calculationg
for model.d['phi'] in np.arange(-0.001, 0.00101, 0.0002):
    Fg_func = lambda x, y, z: disl.Fg(x, y, z) #, filename=saved_Fg_file)
    im, ql, rulers = model.forward(Fg_func)

    # Visualize the simulated image
    if model.d['phi'] == 0.001:
        vlim_im = [0, 1.48]
    else:
        vlim_im = [None, None]
    figax = vis.visualize_im_qi(forward_dict, im, None, rulers, show=False, vlim_im=vlim_im)
    # figax[0].savefig('rocking_curve_phi%.4f.png'%model.d['phi'])
    # plt.close()

# Visualize the reciprocal space wave vector ql
# figax = vis.visualize_im_qi(forward_dict, None, ql, rulers, vlim_qi=[-1e-4, 1e-4])

# Visualize the observation points
lb, ub = -L/2, L/2                                  # in unit of b
extent = np.multiply(bmag*1e6, [lb, ub, lb, ub, lb, ub]) # in the unit of um
fig, ax = vis.visualize_disl_network(disl.d, rn, links, extent=extent, unit='um', show=False)
nskip = 10
r_obs = np.load(saved_Fg_file)['r_obs']*1e6 # in the unit of um
ax.plot(r_obs[::nskip, 0], r_obs[::nskip, 1], r_obs[::nskip, 2],  'C0.', markersize=0.01)
ax.axis('equal')
plt.show()
