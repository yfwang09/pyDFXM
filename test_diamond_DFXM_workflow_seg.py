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
#  casename = 'diamond_MD50000_174x101x100'
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
print('Number of segments:', disl.links.shape[0])
print('Number of dislocations:', len(disl_list))

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

phi = chi = 0
nseg = disl.links.shape[0]
Fg_path = os.path.join(datapath, 'Fg_%s_seg'%casename)
saved_Fg_seg = 'Fg_seg.npz'
if os.path.exists(saved_Fg_seg):
    rawdata = np.load(saved_Fg_seg)
    dint = rawdata['dint']
    Fg_seg = rawdata['Fg_seg']
    rseg = rawdata['rseg']
    nskip = 1
else:
    dint = np.empty(nseg)
    Fg_seg = np.empty(nseg)
    rseg = np.empty(nseg)
    nskip = 1
    for iseg in range(0, nseg, nskip):
        print('segment %d'%iseg)
        saved_Fg_file = os.path.join(Fg_path, 'Fg_iseg%d_phi%.4f_chi%.4f.npz'%(iseg, phi, chi))
        Fg = np.load(saved_Fg_file)['Fg']
        Fg_seg[iseg] = np.abs(Fg).max()
        Fg_func = lambda x, y, z: disl.Fg(x, y, z, filename=saved_Fg_file)
        if os.path.exists(saved_Fg_file):
            im, ql, rulers = model.forward(Fg_func, timeit=True)
        dint[iseg] = im.max() - im.min()

        indA = int(disl.links[iseg][0])
        indB = int(disl.links[iseg][1])
        rA = disl.rn[indA]
        rB = disl.rn[indB]
        ri = np.linalg.norm((rA + rB)/2)
        rseg[iseg] = ri
    np.savez_compressed(saved_Fg_seg, Fg_seg=Fg_seg[::nskip], dint=dint[::nskip], rseg=rseg[::nskip])

#%%
# Visualize the histogram
# np.savez_compressed('Fg_seg', Fg_seg=Fg_seg[::10], dint=dint[::10], rseg=rseg[::10])
# print(Fg_seg[::10], rseg[::10], dint)
fig, ax = plt.subplots()
ax.semilogy(rseg[::nskip]*bmag*1e6, Fg_seg[::nskip], 'o')
plt.show()

fig, ax = plt.subplots()
# ax.hist(dint, bins=100)
ax.plot(rseg[::nskip]*bmag*1e6, dint[::nskip], 'o')
plt.show()


#%%
'''
disl.load_network(config_file, select_seg=[]) # load empty network to calculate the observation points
saved_Fg_file = os.path.join(datapath, 'Fg_%s_DFXM_robs.npz'%casename)
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

disl.load_network(config_file) # load the full network
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
config_ca_inside_file = os.path.join(config_dir, 'config_%s_inside.ca'%casename)
disl.load_network(config_file, select_seg=select_seg_inside)
disl.write_network_ca(config_ca_inside_file, bmag=bmag)
r_obs_xyz_file = os.path.join(datapath, 'r_obs_%s.xyz'%casename)
disl_ca = dio.write_ca(r_obs_xyz_file, np.array([]), np.array([]), obs_cell*1e10, bmag=bmag)

# %% --------------------------------------------------------
# CALCULATE THE DFXM IMAGE
# -----------------------------------------------------------

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

'''
# %%
