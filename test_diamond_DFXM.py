#%%-----------------------------------------------------------------
# DFXM forward calculation for diamond DDD configurations
#
# Developer: Yifan Wang, yfwang09@stanford.edu
# Date: 2023/12/31
#---------------------------------------------------------------

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
casename = 'diamond_10um_60deg_pbc'
# casename = 'diamond_10um_config1_pbc'
# casename = 'diamond_10um_config2_pbc'
# casename = 'diamond_10um_config3_pbc'
# casename = 'diamond_10um_screw_helix1_pbc'
# casename = 'diamond_10um_screw_helix2_pbc'
# casename = 'diamond_10um_screw_helix3_pbc'
# casename = 'diamond_10um_screw_pbc'
# casename = 'diamond_DD0039'
# casename = 'diamond_MD0_200x100x100'
# casename = 'diamond_MD20000_189x100x100'
# casename = 'diamond_MD50000_174x101x100'
# casename = 'diamond_MD100000_149x100x101'
# casename = 'diamond_MD150000_131x100x104'
# casename = 'diamond_MD200000_114x100x107'

config_dir = 'configs'
config_file = os.path.join(config_dir, 'config_%s.vtk'%casename)
config_ca_file = os.path.join(config_dir, 'config_%s.ca'%casename)
config_reduced_ca_file = os.path.join(config_dir, 'config_%s_reduced.ca'%casename)

# Elasticity parameters (Diamond)
input_dict = dgf.default_dispgrad_dict('disl_network')
print(input_dict)

input_dict['nu'] = NU = 0.200       # Poisson's ratio
input_dict['b'] = bmag = 2.522e-10  # Burger's magnitude (m)
two_theta = 48.16                   # 2theta for diamond-(004) (deg)

# Load the dislocation network
disl = dgf.disl_network(input_dict)
disl.load_network(config_file)

# Write the dislocation network into a CA file
ca_data = disl.write_network_ca(config_ca_file, bmag=bmag)
ca_data = disl.write_network_ca(config_reduced_ca_file, bmag=bmag, reduced=True)
disl_list = ca_data['disl_list']
print('Number of dislocations:', len(disl_list))

#%%-------------------------------------------------------
# CALCULATE THE DISPLACEMENT GRADIENT
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
saved_Fg_file = os.path.join(datapath, 'Fg_%s_visualize.npz'%casename)

model = fwd.DFXM_forward(forward_dict, load_res_fn=saved_res_fn)
Ug = model.Ug
print('Ug')
print(Ug)

# L = np.mean(np.diag(disl.cell))
Lx, Ly, Lz = tuple(np.diag(disl.cell))
# lb, ub = -L/2*bmag, L/2*bmag            # in unit of m
lbx, ubx = -Lx/2*bmag, Lx/2*bmag         # in unit of m
lby, uby = -Ly/2*bmag, Ly/2*bmag         # in unit of m
lbz, ubz = -Lz/2*bmag, Lz/2*bmag         # in unit of m
Ngrid = 50
xs = np.linspace(lbx, ubx, Ngrid)         # (Ngrid, )
ys = np.linspace(lby, uby, Ngrid)         # (Ngrid, )
zs = np.linspace(lbz, ubz, Ngrid)         # (Ngrid, )

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
extent = np.multiply(1e6, [lbx, ubx, lby, uby, lbz, ubz]) # in the unit of um
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
im, ql, rulers = model.forward(Fg_func, timeit=True)

# Visualize the simulated image
figax = vis.visualize_im_qi(forward_dict, im, None, rulers) #, vlim_im=[0, 200])

# Visualize the reciprocal space wave vector ql
# figax = vis.visualize_im_qi(forward_dict, None, ql, rulers, vlim_qi=[-1e-4, 1e-4])

# Visualize the observation points
extent = np.multiply(1e6, [lbx, ubx, lby, uby, lbz, ubz]) # in the unit of um
fig, ax = vis.visualize_disl_network(disl.d, disl.rn, disl.links, extent=extent, unit='um', show=False)
nskip = 10
r_obs = np.load(saved_Fg_file)['r_obs']*1e6 # in the unit of um
ax.plot(r_obs[::nskip, 0], r_obs[::nskip, 1], r_obs[::nskip, 2],  'C0.', markersize=0.01)
plt.show()
