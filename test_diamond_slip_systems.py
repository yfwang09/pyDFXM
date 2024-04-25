#%%-----------------------------------------------------------------
# Analysis of dislocation slip systems
#
# Developer: Yifan Wang, yfwang09@stanford.edu
# Date: 2024/04/12
#---------------------------------------------------------------

import os, glob
import numpy as np
import matplotlib.pyplot as plt
import dispgrad_func as dgf
import forward_model as fwd
import visualize_helper as vis
import disl_io_helper as dio
from itertools import product

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
    'diamond_MD50000_174x101x100',
    # 'diamond_MD100000_149x100x101',
    # 'diamond_MD150000_131x100x104',
    # 'diamond_MD200000_114x100x107'
]

for i, casename in enumerate(casenames):
    # if casename == 'diamond_MD0_200x100x100':
    #     test_physical_disl = False
    # else:
    #     test_physical_disl = True

    config_vtk = 'configs'
    config_file = os.path.join(config_vtk, 'config_%s.vtk'%casename)
    config_dir = os.path.join(config_vtk, 'config_%s'%casename)
    os.makedirs(config_dir, exist_ok=True)
    config_ca_file = os.path.join(config_dir, 'config_%s.ca'%casename)
    config_reduced_ca_file = os.path.join(config_dir, 'config_%s_reduced.ca'%casename)
    # config_reduced_ca_file = os.path.join(config_dir, 'config_reduced_%d.ca'%i)

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
    print('CA file saved at %s'%config_ca_file)

    # Combine redundant nodes
    rn, rn_idx = np.unique(disl.rn, axis=0, return_inverse=True)
    links = disl.links.copy()
    links[:, 0:2] = rn_idx[disl.links[:, 0:2].astype(int)]
    disl.rn = disl.d['rn'] = rn
    disl.links = disl.d['links'] = links
    ca_data = disl.write_network_ca(config_reduced_ca_file, bmag=bmag) #, reduced=True)

    # Reduce dislocation segments to physical nodes
    ca_data = disl.write_network_ca('config_reduced.ca', bmag=bmag, reduced=True, pbc=True)
    disl_list = ca_data['disl_list']
    disl_edge_list = ca_data['disl_edge_list']
    print('Number of dislocations:', len(disl_list))

# %%
# merge redundant links
    
links = disl.links.copy()
print('Number of dislocation segments:', links.shape[0])
while True:
    links_distr = np.logical_and(links[:, None, 0] == links[None, :, 1], 
                                 links[:, None, 1] == links[None, :, 0])
    
    reversed_links = np.triu(links_distr, k=1)
    print('Number of reversed links:', np.count_nonzero(reversed_links))
    for i, j in zip(*reversed_links.nonzero()):
        links[j, :2] = links[j, :2][::-1]
        links[j, 2:5] = -links[j, 2:5]
        links[j, 5:8] = -links[j, 5:8]

    links_dist = np.logical_and(links[:, None, 0] == links[None, :, 0], 
                                links[:, None, 1] == links[None, :, 1])

    redundant_links = np.triu(links_dist, k=1)
    print('Number of redundant links:', np.count_nonzero(redundant_links))
    if np.count_nonzero(redundant_links) == 0:
        break
    links_to_be_deleted = redundant_links.nonzero()[1]
    for i, j in zip(*redundant_links.nonzero()):
        rseg = disl.rn[links[i, 1].astype(int)] - disl.rn[links[i, 0].astype(int)]
        rsegnorm = rseg/np.linalg.norm(rseg)
        btot = links[i, 2:5] + links[j, 2:5]
        if np.linalg.norm(btot) < 1e-10:
            links_to_be_deleted = np.append(links_to_be_deleted, i)
        else:
            btotnorm = btot/np.linalg.norm(btot)
            links[i, 2:5] = btotnorm
            links[i, 5:8] = np.cross(btotnorm, rsegnorm)
        # print(i, j, links[i, :2], links[j, :2])
        # print('  ', links[i, 2:5], links[j, 2:5],
        #     links[i, 2:5] + links[j, 2:5])

    links = np.delete(links, links_to_be_deleted, axis=0)

print('Number of dislocation segments:', links.shape[0])
disl.links = disl.d['links'] = links

config_merged_ca_file = os.path.join(config_dir, 'config_%s_merged.ca'%casename)
disl.write_network_ca(config_merged_ca_file, bmag=bmag)

# %%
# Define slip systems

# Slip systems
slip_n = np.array([
    # (111) slip systems
    [1, 1, 1],      # 0: 1/3[111]
    [-1, -1, 1],    # 1: 1/3[-1-11]
    [-1, 1, -1],    # 2: 1/3[-11-1]
    [1, -1, -1],    # 3: 1/3[1-1-1]
    # (001) slip systems
    [0, 0, 1],      # 4: [001]
    [0, 1, 0],      # 5: [010]
    [1, 0, 0],      # 6: [100]
    # (110) slip planes
    [1, 1, 0],      # 7: 1/2[110]
    [1, 0, 1],      # 8: 1/2[101]
    [0, 1, 1],      # 9: 1/2[011]
    [1, -1, 0],     # 10: 1/2[-110]
    [0, 1, -1],     # 11: 1/2[0-11]
    [-1, 0, 1],     # 12: 1/2[-101]
    # (112) slip planes
    [1, 1, 2],      # 13: 1/6[112]
    [1, 2, 1],      # 14: 1/6[121]
    [2, 1, 1],      # 15: 1/6[211]
    [1, 1, -2],     # 16: 1/6[1-12]
    [1, -2, 1],     # 17: 1/6[-121]
    [-2, 1, 1],     # 18: 1/6[-211]
    [1, -1, 2],     # 19: 1/6[1-21]
    [2, 1, -1],     # 20: 1/6[21-1]
    [-1, 2, 1],     # 21: 1/6[-211]
    [-1, 1, 2],     # 22: 1/6[-121]
    [2, -1, 1],     # 23: 1/6[211]
    [1, 2, -1],     # 24: 1/6[121]
])
slip_b = np.array([
    # <110>
    [1, -1, 0],
    [-1, 0, 1],
    [0, 1, -1],
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1],
    # <111>
    [1, 1, 1],
    [-1, 1, 1],
    [1, -1, 1],
    [1, 1, -1],
    # <112>
    [1, 1, 2],
    [1, 2, 1],
    [2, 1, 1],
    [1, 1, -2],
    [1, -2, 1],
    [-2, 1, 1],
    [1, -1, 2],
    [2, 1, -1],
    [-1, 2, 1],
    [-1, 1, 2],
    [2, -1, 1],
    [1, 2, -1],
    # <100>
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
])

# Slip system indices

slip_systems = np.array(np.nonzero(np.dot(slip_n, slip_b.T) == 0), dtype=int).T
print(slip_systems.shape)

# %%
# Test the slip systems for each dislocation segment

# Initialize the slip system list
slip_list = -1001*np.ones(links.shape[0], dtype=int)
# slip systems with well defined slip planes and Burgers vectors
slip_seg = [[] for i in range(slip_systems.shape[0])]
slip_count = np.zeros(slip_systems.shape[0], dtype=int)
slip_len = np.zeros(slip_systems.shape[0])
# slip systems with only well defined Burgers vectors
slip_seg_b = [[] for i in range(slip_b.shape[0])]
slip_count_b = np.zeros(slip_b.shape[0], dtype=int)
slip_len_b = np.zeros(slip_b.shape[0])
# all other slip systems
slip_seg_other = []
slip_count_other = 0
slip_len_other = 0
rtol = 0.1
rtol_b = 1e-3

debug_single = False
debug_all = False

# Test each dislocation segment
for id, link in enumerate(links):
    # line vector
    rseg = rn[link[1].astype(int)] - rn[link[0].astype(int)]
    rsegnorm = rseg/np.linalg.norm(rseg)
    # Burger's vector
    bdots = np.abs(slip_b.dot(link[2:5])/np.linalg.norm(slip_b, axis=-1))
    # print(bdots)
    ind_b = np.isclose(bdots, 1, rtol=rtol_b).nonzero()[0]

    # Slip plane normal
    # a list of normal vectors perpendicular to b
    ndotb = np.abs(slip_n.dot(link[2:5])/np.linalg.norm(slip_n, axis=-1))
    # Find the normal vector perpendicular to the line vector
    ndotr = np.abs(slip_n.dot(rsegnorm)/np.linalg.norm(slip_n, axis=-1))
    ind_n = np.nonzero(np.logical_and(ndotr < rtol, ndotb < rtol_b))[0]
    if id in [923, 919] and debug_single:
        print('Disl segment', id)
        print('  rsegnorm', rsegnorm)
        print("  Burger's vector", link[2:5])
        print("  Normal vector",   np.cross(link[2:5], rsegnorm))
        print(ndotr)
        print(ndotb)
        print((ndotr < rtol).nonzero())
        print((ndotb < rtol_b).nonzero())
        print(ind_n)

    if ind_b.size > 0 and ind_n.size > 0:
        # print(ind_b, ind_n)
        ind_s = np.where((np.isin(slip_systems[:, 0], ind_n)) & (slip_systems[:, 1]==ind_b))[0]
        if ind_s.size > 0:
            for ind_s_item in ind_s[:1]:
                slip_list[id] = ind_s_item
                slip_seg[ind_s_item].append(id)
                slip_count[ind_s_item] += 1
                slip_len[ind_s_item] += np.linalg.norm(rseg)
            continue
    
    if ind_b.size > 0:
        slip_list[id] = - (ind_b.item() + 1)
        slip_seg_b[ind_b.item()].append(id)
        slip_count_b[ind_b.item()] += 1
        slip_len_b[ind_b.item()] += np.linalg.norm(rseg)
        continue
    
    # other slip systems
    print('Disl %d: Slip system not found'%(id, ))
    print('rsegnorm', rsegnorm)
    print("Burger's vector", link[2:5])
    print("Normal vector",   np.cross(link[2:5], rsegnorm))
    slip_seg_other.append(id)
    slip_count_other += 1
    slip_len_other += np.linalg.norm(rseg)
    if debug_all:
        print(ndotr)
        print(ndotb)
        print((ndotr < rtol).nonzero())
        print((ndotb < rtol_b).nonzero())
        print(ind_n)

# %%
# Plot the distribution of dislocation segments on each slip system

slip_sys_n = slip_systems[:, 0]
ind_slip_n = [# np.logical_and(slip_sys_n >= 0, slip_sys_n < 4),      # {111}
            slip_sys_n == 0,            # (111)
            slip_sys_n == 1,            # (-1-11)
            slip_sys_n == 2,            # (-11-1)
            slip_sys_n == 3,            # (1-1-1)
            np.logical_and(slip_sys_n >= 4, slip_sys_n < 7),      # {001}
            np.logical_and(slip_sys_n >= 7, slip_sys_n < 13),     # {110}
            np.logical_and(slip_sys_n >= 13, slip_sys_n < 25),    # {112}
            ]
# label_n = [r'(111)', r'(001)', r'(110)', r'(112)']
label_n = [r'(111)', r'(1-11)', r'(-11-1)', r'(1-1-1)', r'(001)', r'(110)', r'(112)']

slip_sys_b = slip_systems[:, 1]
ind_slip_b = [# np.logical_and(slip_sys_b >= 0, slip_sys_b < 6),      # <110>
            slip_sys_b == 0,            # <1-10>
            slip_sys_b == 1,            # <-101>
            slip_sys_b == 2,            # <01-1>
            slip_sys_b == 3,            # <110>
            slip_sys_b == 4,            # <011>
            slip_sys_b == 5,            # <101>
            np.logical_and(slip_sys_b >= 6, slip_sys_b < 10),     # <111>
            np.logical_and(slip_sys_b >= 10, slip_sys_b < 22),    # <112>
            np.logical_and(slip_sys_b >= 22, slip_sys_b < 25),    # <100>
            ]
# label_b = [r'[110]', r'[111]', r'[112]', r'[100]']
label_b = [r'[1-10]', r'[-101]', r'[01-1]', r'[110]', r'[011]', r'[101]', r'[111]', r'[112]', r'[100]']

other_slip_b = np.arange(slip_b.shape[0], dtype=int)
ind_other_b = [np.logical_and(other_slip_b >= 0, other_slip_b < 6),      # <110>
               np.logical_and(other_slip_b >= 6, other_slip_b < 10),     # <111>
               np.logical_and(other_slip_b >= 10, other_slip_b < 22),    # <112>
               np.logical_and(other_slip_b >= 22, other_slip_b < 25),    # <100>
               ]

# summarize the slip systems
slip_seg_s = []
slip_count_s = []
slip_len_s = []
slip_label_s = []
for i, j in product(range(len(ind_slip_n)), range(len(ind_slip_b))):
    indn = ind_slip_n[i]
    indb = ind_slip_b[j]
    inds = np.logical_and(indn, indb).nonzero()[0]
    slip_seg_s.append(np.isin(slip_list, inds).nonzero()[0])
    slip_count_s.append(np.sum(slip_count[inds]))
    slip_len_s.append(np.sum(slip_len[inds]))
    slip_label_s.append(f'n{label_n[i]}b{label_b[j]}')
for i in range(len(ind_other_b)):
    inds = ind_other_b[i].nonzero()[0]
    slip_seg_s.append(np.isin(slip_list, -inds-1).nonzero()[0])
    slip_count_s.append(np.sum(slip_count_b[inds]))
    slip_len_s.append(np.sum(slip_len_b[inds]))
    slip_label_s.append(f'other b{label_b[i]}')
slip_seg_s.append(np.nonzero(slip_list == -1001)[0])
slip_count_s = np.array(slip_count_s + [slip_count_other, ])
slip_len_s = np.array(slip_len_s + [slip_len_other, ])
slip_label_s = np.array(slip_label_s + ['others', ], dtype=str)
ind_nonzero = np.nonzero(slip_count_s)[0]

# plot all the slip systems
plot_all = False
if plot_all:
    slip_count_s = slip_count
    slip_len_s = slip_len
    slip_label_s = np.array([f'{slip_n[n]}{slip_b[b]}' for n, b in slip_systems] + ['others', ], dtype=str)
    ind_nonzero = np.arange(slip_count_s.size)
    ind_nonzero = ind_nonzero[slip_count_s > 0]

# Plot the distribution of dislocation segments on each slip plane
fs = 18
fig, (ax1, ax) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
xvals = np.arange(ind_nonzero.size)
ax1.bar(xvals, slip_len_s[ind_nonzero])
ax1.set_ylabel('L/b', fontsize=fs)
ax.bar(xvals, slip_count_s[ind_nonzero])
ax.set_xticks(xvals)
ax.set_xticklabels(slip_label_s[ind_nonzero], rotation=90, fontsize=fs)
ax.set_ylabel(r'$n_{\rm seg}$', fontsize=fs)
fig.tight_layout()

# Save the figure
fig.savefig(os.path.join(config_dir, 'slip_systems.png'))
# plt.show()

# %%
# Save the ca files for each slip system groups

rn_processed = disl.rn.copy()
links_processed = disl.links.copy()

f = open(os.path.join(config_dir, 'slip_systems.txt'), 'w')
k = 0
for i in range(len(slip_label_s)):
    if slip_count_s[i] > 0:
        # disl.load_network(config_file, select_seg=slip_seg_s[i])
        disl.links = links_processed[slip_seg_s[i]]
        rn_idx, inverse = np.unique(disl.links[:, :2].astype(int), return_inverse=True)
        disl.rn = rn_processed[rn_idx]
        disl.links[:, :2] = inverse.reshape(-1, 2)

        config_slip_ca_file = os.path.join(config_dir, 'config_%s_slip%d.ca'%(casename, k))
        ca_data = disl.write_network_ca(config_slip_ca_file, bmag=bmag)
        config_slip_vtk_file = os.path.join(config_dir, 'config_%s_slip%d.vtk'%(casename, k))
        btype = np.ones(disl.links.shape[0], dtype=int)*i
        dio.write_vtk(config_slip_vtk_file, disl.rn, disl.links, disl.cell, btype)
        print('Slip system %d'%k, f'{slip_label_s[i]}: {slip_count_s[i]} segs, {slip_len_s[i]:.2f} b')
        print('Slip system %d'%k, f'{slip_label_s[i]}: {slip_count_s[i]} segs, {slip_len_s[i]:.2f} b', file=f)
        k += 1

f.close()
plt.show()