#%%-----------------------------------------------------------------
# DFXM forward calculation for diamond DDD configurations
#
# Developer: Yifan Wang, yfwang09@stanford.edu
# Date: 2023/12/31
#---------------------------------------------------------------

import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import dispgrad_func as dgf
import forward_model as fwd
import visualize_helper as vis
from itertools import product
from mpi4py import MPI

#%%-------------------------------------------------------
# INITIALIZATION
#---------------------------------------------------------

# Configuration files
casename = 'diamond_10um_60deg_pbc'
if len(sys.argv) > 1:
    casename = sys.argv[1]

scale_cell = 1.0/4

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

# Elasticity parameters (Diamond)
input_dict = dgf.default_dispgrad_dict('disl_network')
print(input_dict)

input_dict['nu'] = NU = 0.200       # Poisson's ratio
input_dict['b'] = bmag = 2.522e-10  # Burger's magnitude (m)
two_theta = 48.16                   # 2theta for diamond-(004) (deg)

# Load the dislocation network
disl = dgf.disl_network(input_dict)
# disl.load_network(config_file)
with open(config_file, 'r') as f:
    for line in f:
        if line.startswith('CELLS'):
            nLinks = int(line.split()[1]) - 1
            break

print('Edges:', nLinks)

# Set up the pre-calculated resolution function
datapath = 'data'
os.makedirs(datapath, exist_ok=True)
saved_res_fn = os.path.join(datapath, 'Res_qi_diamond_001.npz')
print('saved resolution function at %s'%saved_res_fn)

Fg_path = os.path.join(datapath, 'Fg_%s_seg'%casename)
os.makedirs(Fg_path, exist_ok=True)
im_path = os.path.join(datapath, 'im_%s_seg'%casename)
os.makedirs(im_path, exist_ok=True)

#%%-------------------------------------------------------
# Set up the MPI environment
#---------------------------------------------------------

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print('The %d-%dth worker is initiated.'%(rank, size))

disl_nseg = np.arange(nLinks, dtype=int)
phi_values = np.linspace(-0.001, 0.001, 21).round(4)
# phi_values = np.array([0.0, ])
chi_values = np.linspace(-0.001, 0.001, 21).round(4)
# chi_values = np.array([0.0, ])

job_content = list(product(disl_nseg, phi_values, chi_values))
numjobs = len(job_content)

# Arrange the works and jobs
if rank == 0:
    # This is head worker
    # Jobs are arranged by this worker
    print('Total number of jobs: %d'%numjobs)
    job_all_idx = np.arange(numjobs, dtype='i')
    # Shuffle the job index to make all workers equal
    # for unbalanced jobs
    # np.random.shuffle(job_all_idx)
    # print(job_all_idx)
else:
    job_all_idx = np.empty(numjobs, dtype='i')

comm.Bcast(job_all_idx, root=0)

# print(job_all_idx)
njob_per_worker = np.ceil(numjobs/size).astype(int)
# The number of jobs should be a multiple of the NumProcess[MPI]

start_idx = rank*njob_per_worker
end_idx = np.min([(rank+1)*njob_per_worker, numjobs])
this_worker_job = job_all_idx[start_idx:end_idx]
work_content = [job_content[x] for x in this_worker_job]

#%% -------------------------------------------------------
# Set up the forward model 
#---------------------------------------------------------
forward_dict = fwd.default_forward_dict()
forward_dict['two_theta'] = two_theta
print(forward_dict)

model = fwd.DFXM_forward(forward_dict, load_res_fn=saved_res_fn)
Ug = model.Ug

# print('Ug')
# print(Ug)

#%%-------------------------------------------------------
# CALCULATE THE DFXM IMAGE
#---------------------------------------------------------

for a_piece_of_work in work_content:
    # print('#'*20 + ' Calculate and visualize the image')
    # print(a_piece_of_work)
    iseg, phi, chi = a_piece_of_work
    print('Worker %d is calculating: iseg=%d, phi=%g, chi=%g'%(rank, iseg, phi, chi), end=';')
    
    # set up the work content
    disl.load_network(config_file, scale_cell=scale_cell, select_seg=[iseg, ])
    model.d['phi'] = phi
    model.d['chi'] = chi
    # L = np.mean(np.diag(disl.cell))
    Lx, Ly, Lz = tuple(np.diag(disl.cell))

    saved_Fg_file = os.path.join(Fg_path, 'Fg_iseg%d_phi%.4f_chi%.4f.npz'%(iseg, phi, chi))
    im_qi_file = os.path.join(im_path, 'im_iseg%d_phi%.4f_chi%.4f.npz'%(iseg, phi, chi))
    # print('saved displacement gradient at %s'%saved_Fg_file)
    Fg_func = lambda x, y, z: disl.Fg(x, y, z, filename=saved_Fg_file)

    tic = time.time()
    # if not os.path.exists(im_qi_file):
    try:
        Fg = np.load(saved_Fg_file)['Fg']
    except:
        if os.path.exists(saved_Fg_file):
            os.remove(saved_Fg_file)
        im, ql, rulers = model.forward(Fg_func) #, timeit=True)
        ruler_x, ruler_y, ruler_z = rulers
        # np.savez_compressed(im_qi_file, im=im, ql=ql, ruler_x=ruler_x, ruler_y=ruler_y, ruler_z=ruler_z)

        # Visualize the simulated image
        # savefile = os.path.join(im_path, 'dfxm_im_iseg%d_phi%.4f_chi%.4f.png'%(iseg, phi, chi))
        # if not os.path.exists(savefile):
        #     figax = vis.visualize_im_qi(forward_dict, im, None, rulers, show=False) #, vlim_im=[0, 200])
        #     fig, ax = figax[0], figax[1]
        #     fig.savefig(savefile)
        #     plt.close()

        # Visualize the reciprocal space wave vector ql
        # figax = vis.visualize_im_qi(forward_dict, None, ql, rulers, vlim_qi=[-1e-4, 1e-4])

    # Visualize the observation points
    savefile = os.path.join(im_path, 'disl_im_iseg%d.png'%(iseg, ))
    if not os.path.exists(savefile):
        lbx, ubx, lby, uby, lbz, ubz = -Lx/2, Lx/2, -Ly/2, Ly/2, -Lz/2, Lz/2 # in unit of b
        extent = np.multiply(bmag*1e6, [lbx, ubx, lby, uby, lbz, ubz]) # in the unit of um
        fig, ax = vis.visualize_disl_network(disl.d, disl.rn, disl.links, extent=extent, unit='um', show=False)
        nskip = 10
        r_obs = np.load(saved_Fg_file)['r_obs']*1e6 # in the unit of um
        ax.plot(r_obs[::nskip, 0], r_obs[::nskip, 1], r_obs[::nskip, 2],  'C0.', markersize=0.01)

        fig.savefig(savefile)
        plt.close()

    toc = time.time()
    print(' takes %.2f seconds.'%(toc-tic))
