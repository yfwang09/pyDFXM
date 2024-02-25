#%%-----------------------------------------------------------------
# DFXM forward calculation for diamond DDD configurations
#
# Developer: Yifan Wang, yfwang09@stanford.edu
# Date: 2024/01/15
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
# casename = 'diamond_10um_60deg_pbc'
# casename = 'diamond_10um_config1_pbc'
# casename = 'diamond_10um_config2_pbc'
# casename = 'diamond_10um_config3_pbc'
# casename = 'diamond_10um_screw_helix1_pbc'
# casename = 'diamond_10um_screw_helix2_pbc'
# casename = 'diamond_10um_screw_helix3_pbc'
# casename = 'diamond_10um_screw_pbc'
# casename = 'diamond_DD0039'
# casename = 'diamond_MD0_200x100x100'
casename = 'diamond_MD20000_189x100x100'
# casename = 'diamond_MD50000_174x101x100'
# casename = 'diamond_MD100000_149x100x101'
# casename = 'diamond_MD150000_131x100x104'
# casename = 'diamond_MD200000_114x100x107'
scale_cell = 1/2                    # scale the cell side by this scale (default = 1)
casename_scaled = casename + '_scale%d'%(1/scale_cell)

config_dir = 'configs'
config_file = os.path.join(config_dir, 'config_%s.vtk'%casename)
config_ca_file = os.path.join(config_dir, 'config_%s.ca'%casename_scaled)
config_reduced_ca_file = os.path.join(config_dir, 'config_%s_reduced.ca'%casename_scaled)

# Elasticity parameters (Diamond)
input_dict = dgf.default_dispgrad_dict('disl_network')
print(input_dict)

input_dict['nu'] = NU = 0.200       # Poisson's ratio
input_dict['b'] = bmag = 2.522e-10  # Burger's magnitude (m)
two_theta = 48.16                   # 2theta for diamond-(004) (deg)

# Load the dislocation network
disl = dgf.disl_network(input_dict)
disl.load_network(config_file, scale_cell=scale_cell)

# Write the dislocation network into a CA file
ca_data = disl.write_network_ca(config_ca_file, bmag=bmag)
ca_data = disl.write_network_ca(config_reduced_ca_file, bmag=bmag, reduced=True)
disl_list = ca_data['disl_list']
print('Number of segments:', disl.links.shape[0])
print('Number of dislocations:', len(disl_list))

#%%-------------------------------------------------------
# SETUP THE FORWARD MODEL
#---------------------------------------------------------

forward_dict = fwd.default_forward_dict()
forward_dict['two_theta'] = two_theta

forward_dict['hkl'] = [ 1, 1, 1]
forward_dict['x_c'] = [ 1, 1,-2]
forward_dict['y_c'] = [-1, 1, 0]

print(forward_dict)

# Set up the pre-calculated resolution function
datapath = 'data'
os.makedirs(datapath, exist_ok=True)
saved_res_fn = os.path.join(datapath, 'Res_qi_diamond_001.npz')
print('saved resolution function at %s'%saved_res_fn)

# Set up the pre-calculated displacement gradient
model = fwd.DFXM_forward(forward_dict, load_res_fn=saved_res_fn)
Ug = model.Ug
print('U matrix in Eq.(7): rs = U.rg (Poulsen et al., 2021)')
print(Ug)

#%%-------------------------------------------------------
# CALCULATE THE OBSERVATION POINTS OF THE DFXM
#---------------------------------------------------------

disl.load_network(config_file, select_seg=[], scale_cell=scale_cell) # load empty network to calculate the observation points
saved_Fg_file = os.path.join(datapath, 'Fg_%s_hkl%d%d%d_DFXM_robs.npz'%((casename, ) + tuple(forward_dict['hkl'])))
print('saved observation points at %s'%saved_Fg_file)
Fg_func = lambda x, y, z: disl.Fg(x, y, z, filename=saved_Fg_file)
if not os.path.exists(saved_Fg_file):
    im, ql, rulers = model.forward(Fg_func, timeit=True)

# Visualize the simulated image
# figax = vis.visualize_im_qi(forward_dict, im, None, rulers) #, vlim_im=[0, 200])

# Visualize the reciprocal space wave vector ql
# figax = vis.visualize_im_qi(forward_dict, None, ql, rulers, vlim_qi=[-1e-4, 1e-4])

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
ax.view_init(azim=90, elev=0)
plt.show()

# %%-------------------------------------------------------
# FILTER THE DISLOCATION SEGMENTS THAT ARE OUTSIDE THE OBSERVATION REGION
#---------------------------------------------------------

# obtain the 
# Load the dislocation network
Nobs = 2
NNxyz = np.multiply(model.d['Npixels'], Nobs)
# NNxyz = (100, 80, 90)
NNxyz[1], NNxyz[2] = NNxyz[2], NNxyz[1]
NNxyz = tuple(NNxyz)
print(NNxyz)
r_obs = np.load(saved_Fg_file)['r_obs']
r_obs_cell = np.swapaxes(np.reshape(r_obs, NNxyz + (3, ), order='F'), 1, 2)
obs_cell = np.transpose([r_obs_cell[-1, 0, 0, :] - r_obs_cell[0, 0, 0, :], 
                         r_obs_cell[0, -1, 0, :] - r_obs_cell[0, 0, 0, :],
                         r_obs_cell[0, 0, -1, :] - r_obs_cell[0, 0, 0, :]
                        ])
print(obs_cell)

# fig = plt.figure(figsize=(12, 12))
# ax = fig.add_subplot(111, projection='3d')
# nskip = 1
# ax.plot(r_obs[::nskip, 0], r_obs[::nskip, 1], r_obs[::nskip, 2], '.', markersize=0.01)
# ax.plot(r_obs_cell[:, 0, 0, 0], r_obs_cell[:, 0, 0, 1], r_obs_cell[:, 0, 0, 2], '-')
# ax.plot(r_obs_cell[0, :, 0, 0], r_obs_cell[0, :, 0, 1], r_obs_cell[0, :, 0, 2], '-')
# ax.plot(r_obs_cell[0, 0, :, 0], r_obs_cell[0, 0, :, 1], r_obs_cell[0, 0, :, 2], '-')
# ax.view_init(elev=0, azim=-90)
# plt.show()
# fig = plt.figure(figsize=(12, 12))
# ax = fig.add_subplot(111, projection='3d')
# nskip = 1
# ax.plot(r_obs[::nskip, 0], r_obs[::nskip, 1], r_obs[::nskip, 2], '.', markersize=0.01)
# ax.plot(r_obs_cell[:, 0, 0, 0], r_obs_cell[:, 0, 0, 1], r_obs_cell[:, 0, 0, 2], '-')
# ax.plot(r_obs_cell[0, :, 0, 0], r_obs_cell[0, :, 0, 1], r_obs_cell[0, :, 0, 2], '-')
# ax.plot(r_obs_cell[0, 0, :, 0], r_obs_cell[0, 0, :, 1], r_obs_cell[0, 0, :, 2], '-')
# ax.view_init(elev=0, azim=-90)
# plt.show()

disl.load_network(config_file, scale_cell=scale_cell) # load the full network
nsegs = disl.links.shape[0]
select_seg_inside = []
for ilink in range(nsegs):
    link = disl.links[ilink]
    end1 = disl.rn[int(link[0])]*bmag
    end2 = disl.rn[int(link[1])]*bmag
    s1 = np.linalg.inv(obs_cell).dot(end1)
    s2 = np.linalg.inv(obs_cell).dot(end2)
    if np.all(np.abs(s1) < 0.51) or np.all(np.abs(s2) < 0.51):
        select_seg_inside.append(ilink)
print('# of segments inside the observation region:', len(select_seg_inside))

# %% --------------------------------------------------------
# SAVE THE DISLOCATION NETWORK INSIDE THE OBSERVATION REGION
# -----------------------------------------------------------

import disl_io_helper as dio
config_ca_inside_file = os.path.join(config_dir, 'config_%s_inside.ca'%casename_scaled)
disl.load_network(config_file, select_seg=select_seg_inside, scale_cell=scale_cell)
ca_data = disl.write_network_ca(config_ca_inside_file, bmag=bmag)

# %% --------------------------------------------------------
# CALCULATE THE DFXM IMAGE
# -----------------------------------------------------------

print('#'*20 + ' Calculate and visualize the image')
saved_Fg_file = os.path.join(datapath, 'Fg_%s_DFXM.npz'%casename_scaled)
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

# %%

r_obs_xyz_file = os.path.join(datapath, 'r_obs_%s.xyz'%casename_scaled)
im_Nobs = np.repeat(np.repeat(im, Nobs, axis=0), Nobs, axis=1)[:,:,np.newaxis]
im_obs = np.tile(im_Nobs, (1, 1, model.d['Npixels'][2]*Nobs)).reshape((-1, 1))
r_obs = r_obs_cell.reshape((-1, 3))
dio.write_xyz(r_obs_xyz_file, r_obs, props=im_obs, scale=1e10)

# %%

print('#'*20 + ' Calculate and visualize the image')
cell_scale = 1/4
disl.load_network(config_file, scale_cell=cell_scale) # load the full network
print('load full network with scale 1/%d'%(1/cell_scale))

output_path = os.path.join(datapath, 'output_%s_scale%d'%(casename, 1/cell_scale))
import glob
Fg_files = glob.glob(os.path.join(output_path, 'Fg_*.npz'))

saved_Fg_file = os.path.join(output_path, 'Fg_%s_DFXM.npz'%casename)
if not os.path.exists(saved_Fg_file):
    Fg = None
    for Fg_file in Fg_files:
        print('saved displacement gradient at %s'%saved_Fg_file)
        if Fg is None:
            Fg = np.load(Fg_file)['Fg']
            r_obs = np.load(Fg_file)['r_obs']
        else:
            try:
                Fg += np.load(Fg_file)['Fg']
            except:
                Fg += 0

    np.savez_compressed(saved_Fg_file, Fg=Fg, r_obs=r_obs)
else:
    Fg = np.load(saved_Fg_file)['Fg']
    r_obs = np.load(saved_Fg_file)['r_obs']

print('saved displacement gradient at %s'%saved_Fg_file)
Fg_func = lambda x, y, z: disl.Fg(x, y, z, filename=saved_Fg_file)
im, ql, rulers = model.forward(Fg_func, timeit=True)

# Visualize the simulated image
figax = vis.visualize_im_qi(forward_dict, im, None, rulers) #, vlim_im=[0, 200])

# Visualize the reciprocal space wave vector ql
# figax = vis.visualize_im_qi(forward_dict, None, ql, rulers, vlim_qi=[-1e-4, 1e-4])

# %%
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
# %%
r_obs_xyz_file = os.path.join(output_path, 'r_obs_%s.xyz'%casename)
im_Nobs = np.repeat(np.repeat(im, Nobs, axis=0), Nobs, axis=1)[:,:,np.newaxis]
im_obs = np.tile(im_Nobs, (1, 1, model.d['Npixels'][2]*Nobs)).reshape((-1, 1))
r_obs = r_obs_cell.reshape((-1, 3))
dio.write_xyz(r_obs_xyz_file, r_obs, props=im_obs, scale=1e10) # Angstrom
# %%
