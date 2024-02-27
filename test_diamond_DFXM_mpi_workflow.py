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

from mpi4py import MPI

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
scale_cell = 1                    # scale the cell side by this scale (default = 1)
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

# Diffraction plane of diamond (004)
two_theta = 48.16                   # 2theta for diamond-(004) (deg)
hkl = [0, 0, 1]                     # hkl for diamond-(004) plane
x_c = [1, 0, 0]                     # x_c for diamond-(004) plane
y_c = [0, 1, 0]                     # y_c for diamond-(004) plane

# Diffraction plane of diamond (111)
# two_theta = 20.06                   # 2theta for diamond-(111) (deg)
# hkl = [1, 1, 1]                     # hkl for diamond-(111) plane
# x_c = [1, 1, -2]                    # x_c for diamond-(111) plane
# y_c = [-1, 1, 0]                    # y_c for diamond-(111) plane

casename_scaled_hkl = casename_scaled + '_hkl%d%d%d'%tuple(hkl)

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
forward_dict['hkl'] = hkl
forward_dict['x_c'] = x_c
forward_dict['y_c'] = y_c

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
saved_Fg_file = os.path.join(datapath, 'Fg_%s_DFXM_robs.npz'%casename_scaled_hkl)
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
# fig, ax = vis.visualize_disl_network(disl.d, disl.rn, disl.links, extent=extent, unit='um', show=False)
# nskip = 10
# r_obs = np.load(saved_Fg_file)['r_obs']*1e6 # in the unit of um
# ax.plot(r_obs[::nskip, 0], r_obs[::nskip, 1], r_obs[::nskip, 2],  'C0.', markersize=0.01)
# ax.view_init(azim=90, elev=0)
# plt.show()

# %%-------------------------------------------------------
# FILTER THE DISLOCATION SEGMENTS THAT ARE OUTSIDE THE OBSERVATION REGION
#---------------------------------------------------------

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
config_ca_inside_file = os.path.join(config_dir, 'config_%s_inside.ca'%casename_scaled_hkl)
disl.load_network(config_file, select_seg=select_seg_inside, scale_cell=scale_cell)
if not os.path.exists(config_ca_inside_file):
    ca_data = disl.write_network_ca(config_ca_inside_file, bmag=bmag)

# %% --------------------------------------------------------
# CALCULATE THE DISPLACEMENT GRADIENT
# -----------------------------------------------------------

import displacement_grad_helper as dgh

saved_empty_Fg_file = os.path.join(datapath, 'Fg_%s_DFXM_robs.npz'%casename_scaled_hkl)
r_obs = np.load(saved_empty_Fg_file)['r_obs']
rnorm = r_obs/disl.d['b']

rn = disl.rn
links = disl.links

verbose = True

tic_all = time.time()

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
if verbose:
    print('The %d-%dth worker is initiated.'%(rank, size))

dudx_worker = np.zeros((r_obs.shape[0], 3, 3), dtype=float)
for i in range(rank, links.shape[0], size):
    # if verbose:
    #     print('worker %d: calculating link %d'%(rank, i))
    n1 = int(links[i, 0])
    n2 = int(links[i, 1])
    r1 = rn[n1, :]
    r2 = rn[n2, :]
    b = links[i, 2:5]
    tic = time.time()
    dudx_worker += dgh.displacement_gradient_seg_optimized(disl.d['nu'], b, r1, r2, rnorm, disl.a)
    toc = time.time()
    if verbose:
        print('worker %d: link %d is calculated in %.2f seconds'%(rank, i, toc-tic))


if rank == 0:
    dudx_receive = np.empty((size, r_obs.shape[0], 3, 3), dtype=float)
else:
    dudx_receive = None

comm.Gather(dudx_worker, dudx_receive, root=0)

if rank == 0:
    print('#'*20 + ' Calculate and visualize the image')
    saved_Fg_file = os.path.join(datapath, 'Fg_%s_DFXM.npz'%casename_scaled_hkl)
    print('saved displacement gradient at %s'%saved_Fg_file)
    Fg_list = dudx_receive.sum(axis=0)
    toc_all = time.time()
    print('The displacement gradient is calculated in %.2f seconds'%(toc_all-tic_all))
    np.savez_compressed(saved_Fg_file, Fg=Fg_list, r_obs=r_obs)

# %% --------------------------------------------------------
# CALCULATE THE DFXM IMAGE
# -----------------------------------------------------------

if rank == 0:
    print('#'*20 + ' Calculate and visualize the image')
    saved_Fg_file = os.path.join(datapath, 'Fg_%s_DFXM.npz'%casename_scaled_hkl)
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

    r_obs_xyz_file = os.path.join(datapath, 'r_obs_%s.xyz'%casename_scaled_hkl)
    im_Nobs = np.repeat(np.repeat(im, Nobs, axis=0), Nobs, axis=1)[:,:,np.newaxis]
    im_obs = np.tile(im_Nobs, (1, 1, model.d['Npixels'][2]*Nobs)).reshape((-1, 1))
    r_obs = r_obs_cell.reshape((-1, 3))
    if not os.path.exists(r_obs_xyz_file):
        dio.write_xyz(r_obs_xyz_file, r_obs, props=im_obs, scale=1e10)
