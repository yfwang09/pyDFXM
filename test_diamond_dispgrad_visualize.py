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
# casename = 'diamond_MD20000_189x100x100'
# casename = 'diamond_MD50000_174x101x100'
# casename = 'diamond_MD100000_149x100x101'
# casename = 'diamond_MD150000_131x100x104'
# casename = 'diamond_MD200000_114x100x107'
casenames = [
    # 'diamond_10um_60deg_pbc',
    # 'diamond_10um_config1_pbc',
    # 'diamond_10um_config2_pbc',
    # 'diamond_10um_config3_pbc',
    # 'diamond_10um_screw_helix1_pbc',
    # 'diamond_10um_screw_helix2_pbc',
    # 'diamond_10um_screw_helix3_pbc',
    # 'diamond_10um_screw_pbc',
    # 'diamond_DD0039',
    # 'diamond_MD0_200x100x100', 
    # 'diamond_MD20000_189x100x100',
    # 'diamond_MD50000_174x101x100', 
    'diamond_MD100000_149x100x101', 
    'diamond_MD150000_131x100x104', 
    'diamond_MD200000_114x100x107'
]

for casename in casenames:

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
    Fg_path = r'G:\My Drive\Postdoc\Shock Dislocations\vtkfiles'

    saved_Fg_file = os.path.join(Fg_path, 'Fg_%s_visualize.npz'%casename)
    if not os.path.exists(saved_Fg_file):
        dispgrad_file = os.path.join(Fg_path, 'dispgrad_%s.vtk'%casename)
        with open(dispgrad_file, 'r') as f:
            while True:
                line = f.readline()
                if line == '': break
                if line.startswith('DIMENSIONS'):
                    Ngrids = [int(x) for x in line.split()[1:]]
                if line.startswith('POINTS'):
                    Npoints = int(line.split()[1])
                    Rs = np.zeros((Npoints, 3))
                    Fg = np.zeros((Npoints, 3, 3))
                    for k in range(Npoints):
                        line = f.readline()
                        Rs[k, :] = [float(x) for x in line.split()]
                if line.startswith('SCALARS'):
                    Uij = line.split()[1]
                    i, j = int(Uij[1]) - 1, int(Uij[2]) - 1
                    line = f.readline() # LOOKUP_TABLE default
                    for k in range(Npoints):
                        line = f.readline()
                        Fg[k, i, j] = float(line.split()[0])
                        # if i != j:
                        #     print(k, i, j, Fg[k, i, j])
        # Rs = np.swapaxes(
        #         np.swapaxes(
        #             Rs.reshape(tuple(Ngrids) + (3, )), 0, 2
        #         ), 0, 1
        #     )
        Rs = Rs.reshape(tuple(Ngrids) + (3, ), order='F')
        # Fg = np.swapaxes(
        #         np.swapaxes(
        #             Fg.reshape(tuple(Ngrids) + (3, 3)), 0, 2
        #         ), 0, 1
        #     )
        Fg = Fg.reshape(tuple(Ngrids) + (3, 3), order='F')
        np.savez(saved_Fg_file, r_obs=Rs, Fg=Fg)
        print('saved displacement gradient at %s'%saved_Fg_file)
    else:
        Rs = np.load(saved_Fg_file)['r_obs']
        Fg = np.load(saved_Fg_file)['Fg']
        print('loaded displacement gradient at %s'%saved_Fg_file)

    # print(Fg[..., 1, 2].min(), Fg[..., 1, 2].max())

    model = fwd.DFXM_forward(forward_dict, load_res_fn=saved_res_fn)
    Ug = model.Ug
    print('Ug')
    print(Ug)

    L = np.diag(disl.cell)
    xs = Rs[:, 0, 0, 0] - L[0]/2
    ys = Rs[0, :, 0, 1] - L[1]/2
    zs = Rs[0, 0, :, 2] - L[2]/2
    lbx, ubx = xs.min(), xs.max()
    lby, uby = ys.min(), ys.max()
    lbz, ubz = zs.min(), zs.max()
    print('Grid size in the sample coordinates:', Rs.shape, lbx, ubx, lby, uby, lbz, ubz)

    print('Convert Rs into the grain coordinates (Miller indices)')
    Rg = np.einsum('ij,...j->...i', Ug, Rs)
    tic= time.time()
    Fg = disl.Fg(Rg[..., 0], Rg[..., 1], Rg[..., 2], filename=saved_Fg_file)
    toc = time.time()
    print('Time elapsed for calculating Fg: %.2f s'%(toc-tic))
    # print(Fg.min(), Fg.max())

    print('Visualize the displacement gradient')
    i, j = 1, 2
    vmin, vmax = -1e-4, 1e-4 #Fg[..., i, j].min(), Fg[..., i, j].max()
    # print(vmin, vmax)
    fs = 12
    extent = np.multiply(bmag*1e6, [lbx, ubx, lby, uby, lbz, ubz])
    figax = vis.plot_3d_slice_z(Fg[:, :, :, i, j], extent=extent, vmin=vmin, vmax=vmax, nslice=5, fs=fs, show=False)
    # figax = vis.visualize_disl_network(disl.d, disl.rn, disl.links, unit='um', figax=figax, show=False)
    plt.show()

# #%%-------------------------------------------------------
# # CALCULATE THE DFXM IMAGE
# #---------------------------------------------------------

# print('#'*20 + ' Calculate and visualize the image')
# Fg_path = r'G:\My Drive\Codes\DFXM-Simulation-Code\PythonImplementation'
# saved_Fg_file = os.path.join(Fg_path, 'Fg_files_%s'%casename, 'Fg_%s_DFXM.npz'%casename)
# print('saved displacement gradient at %s'%saved_Fg_file)
# Fg_func = lambda x, y, z: disl.Fg(x, y, z, filename=saved_Fg_file)
# im, ql, rulers = model.forward(Fg_func)

# # Visualize the simulated image
# figax = vis.visualize_im_qi(forward_dict, im, None, rulers) #, vlim_im=[0, 200])

# # Visualize the reciprocal space wave vector ql
# # figax = vis.visualize_im_qi(forward_dict, None, ql, rulers, vlim_qi=[-1e-4, 1e-4])

# # Visualize the observation points
# lb, ub = -L/2, L/2                                  # in unit of b
# extent = np.multiply(bmag*1e6, [lb, ub, lb, ub, lb, ub]) # in the unit of um
# fig, ax = vis.visualize_disl_network(disl.d, disl.rn, disl.links, extent=extent, unit='um', show=False)
# nskip = 10
# r_obs = np.load(saved_Fg_file)['r_obs']*1e6 # in the unit of um
# ax.plot(r_obs[::nskip, 0], r_obs[::nskip, 1], r_obs[::nskip, 2],  'C0.', markersize=0.01)
# plt.show()
