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
    # 'diamond_MD100000_149x100x101',
    # 'diamond_MD150000_131x100x104',
    'diamond_MD200000_114x100x107'
]
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
    # ca_data = disl.write_network_ca(config_reduced_ca_file, bmag=bmag, reduced=True)
    ca_data = disl.write_network_ca('config_reduced.ca', bmag=bmag, reduced=True, pbc=True)
    disl_list = ca_data['disl_list']
    print('CA file saved at %s'%config_ca_file)
    print('Number of dislocations:', len(disl_list))

# %%
disl_lens = np.zeros(len(disl_list))
for i, dislocation in enumerate(disl_list):
    rvec = ca_data['rn'][dislocation[1:], :] - ca_data['rn'][dislocation[:-1], :]
    # print(rvec.shape, len(dislocation))
    # if i == 3:
    #     print(dislocation)
    #     print(disl.rn[dislocation[1], :])
    #     print(disl.rn[dislocation[0], :])
    svec = np.linalg.inv(disl.cell).dot(rvec.T).T
    svec = svec - np.round(svec)
    rvec = disl.cell.dot(svec.T).T
    disl_lens[i] = np.linalg.norm(rvec, axis=1).sum()
# print(disl_lens)

fig, ax = plt.subplots()
unit_conversion = 1e-10*1e6 # A to um

ax.hist(disl_lens*unit_conversion, bins=100)
ax.set_yscale('log')
ax.set_xlabel('Dislocation length (um)')
ax.set_ylabel('Counts')
plt.show()
# %%
