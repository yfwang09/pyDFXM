#%%-----------------------------------------------------------------
# DFXM forward calculation for diamond DDD configurations
#
# Developer: Yifan Wang, yfwang09@stanford.edu
# Date: 2023/12/31
#---------------------------------------------------------------

import os, glob
import numpy as np
import matplotlib.pyplot as plt
import dispgrad_func as dgf
import forward_model as fwd
import visualize_helper as vis

#%%-------------------------------------------------------
# INITIALIZATION
#---------------------------------------------------------

# Configuration files
casenames = ['diamond_10um_60deg_pbc',
             'diamond_10um_config1_pbc',
             'diamond_10um_config2_pbc',
             'diamond_10um_config3_pbc',
             'diamond_10um_screw_helix1_pbc',
             'diamond_10um_screw_helix2_pbc',
             'diamond_10um_screw_helix3_pbc',
             'diamond_10um_screw_pbc',
             'diamond_DD0039',
             'diamond_MD0_200x100x100',
             'diamond_MD20000_189x100x100',
             'diamond_MD50000_174x101x100',
             'diamond_MD100000_149x100x101',
             'diamond_MD150000_131x100x104',
             'diamond_MD200000_114x100x107']
# 1336.11, 706.298, 707.025
# 0.00431674409888924069036372682687

for i, casename in enumerate(casenames):
    config_dir = 'configs'
    config_file = os.path.join(config_dir, 'config_%s.vtk'%casename)
    config_ca_file = os.path.join(config_dir, 'config_%s.ca'%casename)
    # config_reduced_ca_file = os.path.join(config_dir, 'config_%s_reduced.ca'%casename)
    config_reduced_ca_file = os.path.join(config_dir, 'config_reduced_%d.ca'%i)

    # Elasticity parameters (Diamond)
    input_dict = dgf.default_dispgrad_dict('disl_network')
    # print(input_dict)

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
    print('CA file saved at %s'%config_ca_file)
    print('Number of dislocations:', len(disl_list))
