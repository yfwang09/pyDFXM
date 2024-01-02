#%%-----------------------------------------------------------------
# DFXM forward calculation for Aluminum discrete dislocation configuration
#
# Developer: Yifan Wang, yfwang09@stanford.edu
# Date: 2023/12/29
#---------------------------------------------------------------

from calendar import c
import os, time
import numpy as np
import matplotlib.pyplot as plt
import dispgrad_func as dgf
import forward_model as fwd
import visualize_helper as vis

#%%-------------------------------------------------------
# INITIALIZATION
#---------------------------------------------------------

# Configuration files
casename = 'Al_10um_edge_pbc'

config_dir = 'configs'
config_file = os.path.join(config_dir, 'config_%s.vtk'%casename)
config_ca_file = os.path.join(config_dir, 'config_%s.ca'%casename)
config_reduced_ca_file = os.path.join(config_dir, 'config_%s_reduced.ca'%casename)

# Elasticity parameters (Aluminum)
input_dict = dgf.default_dispgrad_dict('disl_network')

input_dict['nu'] = NU = 0.324       # Poisson's ratio
input_dict['b'] = bmag = 2.86e-10   # Burger's magnitude (m)

# Load the dislocation network
disl = dgf.disl_network(input_dict)
disl.load_network(config_file)

# Write the dislocation network into a CA file
disl.write_network_ca(config_ca_file, bmag=bmag)
disl.write_network_ca(config_reduced_ca_file, bmag=bmag, reduced=True)

#%%-------------------------------------------------------
# CALCULATE THE DISPLACEMENT GRADIENT
#---------------------------------------------------------

forward_dict = fwd.default_forward_dict()
forward_dict['two_theta'] = 20.73         # 2theta for Al-(002) (deg)
print(forward_dict)

# Set up the pre-calculated resolution function
datapath = 'data'
os.makedirs(datapath, exist_ok=True)
saved_res_fn = os.path.join(datapath, 'Res_qi_Al_001.npz')
print('saved resolution function at %s'%saved_res_fn)

# Set up the pre-calculated displacement gradient
saved_Fg_file = os.path.join(datapath, 'Fg_%s_visualize.npz'%casename)

model = fwd.DFXM_forward(forward_dict, load_res_fn=saved_res_fn)
Ug = model.Ug
print('Ug')
print(Ug)

L = np.mean(np.diag(disl.cell))
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
tic= time.time()
Fg = disl.Fg(Rg[..., 0], Rg[..., 1], Rg[..., 2], filename=saved_Fg_file)
toc = time.time()
print('Time elapsed for calculating Fg: %.2f s'%(toc-tic))

print('Visualize the displacement gradient')
i, j = 1, 2
vmin, vmax = -1e-4, 1e-4
fs = 12
extent = np.multiply(1e6, [lb, ub, lb, ub, lb, ub]) # in the unit of um
figax = vis.plot_3d_slice_z(Fg[:, :, :, i, j], extent=extent, vmin=vmin, vmax=vmax, nslice=5, fs=fs, show=False)
figax = vis.visualize_disl_network(disl.d, disl.rn, disl.links, unit='um', figax=figax, show=False)
plt.show()

#%%-------------------------------------------------------
# CALCULATE THE DFXM IMAGE
#---------------------------------------------------------

print('#'*20 + ' Calculate and visualize the image')
saved_Fg_file = os.path.join(datapath, 'Fg_%s_DFXM.npz'%casename)
print('saved displacement gradient at %s'%saved_Fg_file)
Fg_func = lambda x, y, z: disl.Fg(x, y, z, filename=saved_Fg_file)
im, ql, rulers = model.forward(Fg_func)

# Visualize the simulated image
figax = vis.visualize_im_qi(forward_dict, im, None, rulers) #, vlim_im=[0, 200])

# Visualize the reciprocal space wave vector ql
# figax = vis.visualize_im_qi(forward_dict, None, ql, rulers, vlim_qi=[-1e-4, 1e-4])

# Visualize the observation points
lb, ub = -L/2*bmag, L/2*bmag                        # in unit of b
extent = np.multiply(1e6, [lb, ub, lb, ub, lb, ub]) # in the unit of um
fig, ax = vis.visualize_disl_network(disl.d, disl.rn, disl.links, extent=extent, unit='um', show=False)
nskip = 10
r_obs = np.load(saved_Fg_file)['r_obs']*1e6 # in the unit of um
ax.plot(r_obs[::nskip, 0], r_obs[::nskip, 1], r_obs[::nskip, 2],  'C0.', markersize=0.01)
plt.show()
