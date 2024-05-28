# This script is used to create the statistics of the slip systems in the MD simulation
# date: 2024/05/24
# author: Yifan Wang (yfwang09@stanford.edu)

#%%
# Importing necessary libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visualize_helper as vis
import dispgrad_func as dgf
import disl_io_helper as dio

#%%
# Define the elasticity properties of diamond
# and the input dictionary for the dislocation network

input_dict = dgf.default_dispgrad_dict('disl_network')

input_dict['nu'] = NU = 0.200       # Poisson's ratio
input_dict['b'] = bmag = 2.522e-10  # Burger's magnitude (m)
two_theta = 48.16                   # 2theta for diamond-(004) (deg)

# Set up the dislocation network object
disl = dgf.disl_network(input_dict)

#%%
# Define the functions

def load_disl_network(casename, verbose=False, 
        config_vtk='configs', select_seg=None,
        save_ca_file=None, reduced=False, scale_cell=1.0
    ):
    config_file = os.path.join(config_vtk, 'config_%s.vtk'%casename)
    config_dir = os.path.join(config_vtk, 'config_%s'%casename)
    os.makedirs(config_dir, exist_ok=True)

    disl.load_network(config_file, select_seg=select_seg, scale_cell=scale_cell)
    if save_ca_file is not None:
        disl.write_network_ca(os.path.join(config_dir, save_ca_file), bmag=bmag, reduced=reduced, pbc=True)

    if verbose:
        print('node number', disl.rn.shape)
        print('link number', disl.links.shape)

    return disl.rn, disl.links

def vec2int(bvecs, atol=1e-5):
    ''' Calculate the integer Burgers vector
    
    Parameters
    ----------
    bvecs : array_like (n, 3)
        Burgers vector

    Returns
    -------
    bints : array_like (n, 3)
        Integer Burgers vector
    bnorm : array_like (n,)
        Integer Burgers magnitude
    '''
    if bvecs.shape == (3, ):
        bvecs.reshape(1, 3)

    # Normalize the Burger's vectors to find the integer notations
    invblen = bvecs**2
    invblen[np.isclose(invblen, 0, atol=atol)] = -1 # filter out zero values
    bnorm = (1/invblen).max(axis=1, keepdims=True).round()
    bints = (bvecs*np.sqrt(bnorm)).round()
    # Reduce the Burger's vectors with different signs to the same sign
    bints[bints[:, 0] < 0, :] *= -1
    bints[np.logical_and(bints[:, 0] == 0, bints[:, 1] < 0), :] *= -1
    bints[np.logical_and(np.logical_and(bints[:, 0] == 0, bints[:, 1] == 0), bints[:, 2] < 0), :] *= -1
    bnorm = np.sum(bints**2, axis=1)

    return bints.astype(int), bnorm.flatten().astype(int)

def unique_vectors(nints, verbose=None):
    nunique, n_inv, n_cnt = np.unique(nints, return_inverse=True, return_counts=True, axis=0)           # (n_unique, 3), (n_nvec, ), (n_unique, )
    n_ind = (n_inv[None, :] == np.arange(n_cnt.size, dtype=int)[:, None])   # (n_unique, n_nvec)
    # Get the unique values of the nnorm
    nnorm = np.sum(nunique**2, axis=1)  # (n_unique, )
    nnorm_unique, nnorm_inv, nnorm_cnt = np.unique(nnorm, return_inverse=True, return_counts=True)              # (n_norm_unique, ), (n_unique, ), (n_norm_unique, )

    for i, nnorm in enumerate(nnorm_unique):
        nprint = nunique[nnorm_inv == i]
        ncount = n_cnt[nnorm_inv == i]
        n_inds = n_ind[nnorm_inv == i, :]
        if verbose is not None:
            print('%s: 1/%d'%(verbose, nnorm), len(nprint))
            for k in range(len(nprint)):
                print('    [%3d%3d%3d] %d,'%(*tuple(nprint[k].astype(int)), ncount[k]))

#%%
# function for calculating the slip systems

def slip_system_analysis(disl, verbose=True):
    nlink = disl.links.shape[0]
    
    # Delete the redundant nodes
    rn_reduced, rn_idx = np.unique(disl.rn, axis=0, return_inverse=True)
    links = disl.links.copy()
    links[:, 0:2] = rn_idx[disl.links[:, 0:2].astype(int)]

    # Calculate the Burgers and Normal vectors
    rvecs = disl.rn[disl.links[:, 1].astype(int), :] - disl.rn[disl.links[:, 0].astype(int), :]
    rnorm = np.linalg.norm(rvecs, axis=1)
    bvecs = disl.links[:, 2:5]
    nvecs = disl.links[:, 5:8]
    # rints, _ = vec2int(rvecs)
    # rname = ['[%3d%3d%3d]'%tuple(rints[k, :]) for k in range(nlink)]
    bints, bnorm = vec2int(bvecs)
    nints, nnorm = vec2int(nvecs, atol=0.02)
    bname = ['1/%d[%3d%3d%3d]'%(bnorm[k], *tuple(bints[k, :])) for k in range(nlink)]
    nname = ['1/%d(%3d%3d%3d)'%(nnorm[k], *tuple(nints[k, :])) for k in range(nlink)]
    sname = ['%s %s'%(bname[k], nname[k]) for k in range(nlink)]
    rbang = np.rad2deg(np.arccos(np.abs(np.sum(rvecs * bvecs, axis=1)/rnorm)))

    # Define the pandas dataframe
    df = pd.DataFrame(
        {
            'r0_id': disl.links[:, 0].astype(int),
            'r1_id': disl.links[:, 1].astype(int),
            'rnorm': rnorm, 
            # 'rname': rname,
            'angle': rbang,
            # 'r0_reduced': links[:, 0].astype(int),
            # 'r1_reduced': links[:, 1].astype(int),
            # 'sname': pd.Categorical(sname),
            'sname': pd.Series(sname, dtype='string'),
            'bnorm': bnorm, #'bname': pd.Categorical(bname),
            # 'bi0': bints[:, 0], 'bi1': bints[:, 1], 'bi2': bints[:, 2],
            'bname': pd.Series(bname, dtype='string'),
            'nnorm': nnorm, #'nname': pd.Categorical(nname), 
            'nname': pd.Series(nname, dtype='string'),
            # 'ni0': nints[:, 0], 'ni1': nints[:, 1], 'ni2': nints[:, 2],
        }
    )
    return df, (rn_reduced, links)

# %%
# Function for slip system statistics

def slip_system_stats(df, verbose=True, 
    conds={'full dislocation': ['bnorm == 2 and nnorm == 3']}
):
    data = {}
    for key, cond in conds.items():
        condition, disl_type_name = tuple(cond)
        subgroup = df.query(condition)
        if disl_type_name == 'sname':
            subgroup = subgroup.assign(disltype=subgroup[disl_type_name])
        elif disl_type_name == 'bname':
            subgroup = subgroup.assign(disltype=subgroup[disl_type_name].str.cat([key,]*subgroup.shape[0], sep=' '))
        else:
            subgroup = subgroup.assign(disltype=key)
        
        sorted = subgroup.sort_values(by='disltype')
        if verbose:
            if sorted.shape[0] > 0:
                print('%6d/%6d    %s'%(sorted.shape[0], df.shape[0], sorted.iat[0, -1]))
            else:
                print('%6d/%6d    %s'%(0, df.shape[0], key))
        data[key] = sorted
    return data

# %%
# Statistics of the slip system

casenames = [
    'diamond_MD0_200x100x100',
    'diamond_MD20000_189x100x100',
    'diamond_MD50000_174x101x100',
    'diamond_MD100000_149x100x101',
    'diamond_MD150000_131x100x104', 
    'diamond_MD200000_114x100x107'
]

disltypes = {
    r'<110>{111} full': # 'full dislocation': 
        ['bnorm == 2 and (nnorm == 3 or angle < 5)', ''],
    r'<111> frank':
        ['bnorm == 3', ''],
    # '1/2<110> jog': # 'full jog':
    #     ['bnorm == 2 and nnorm != 3', ''],
    r'<011>{100}': # 'full jog1':
        ['bnorm == 2 and nnorm == 1 and angle >= 5', ''],
    r'<110>{110}': # 'full jog2':
        ['bnorm == 2 and nnorm == 2 and angle >= 5', ''],
    # r'<110>{311}': # 'full jog3':
    #     ['bnorm == 2 and nnorm == 11', ''],
    r'<110> full jog':
        ['bnorm == 2 and (nnorm > 3 and angle >= 5)', ''],
    # 'stair rod1':
    #     ['bnorm == 1', 'bname'],
    # 'stair rod2':
    #     ['bnorm == 10', 'bname'],
    r'<112>{111} shockley': # 'shockley partial':
        ['bnorm == 6', ''],
    # r'<112>{111} shockley': # 'shockley partial':
    #     ['bnorm == 6 and nnorm == 3', ''],
    # r'<112> shockley jog':
    #     ['bnorm == 6 and nnorm != 3', ''],
    'other':
        ['bnorm != 2 and bnorm != 3 and bnorm != 6', ''],
}

verbose = True
# fig, ax = plt.subplots(figsize=(8, 6))
fs = 16

group_list = []
counts = []
length = []

for icase, casename in enumerate(casenames[:]):
    if verbose:
        print(casename)
    rn, links = load_disl_network(casename)
    df, disl_reduced = slip_system_analysis(disl, verbose=True)

    groups = slip_system_stats(df, verbose=verbose, conds=disltypes)
    group_list.append(groups)
    sorted = pd.concat([groups[key] for key in groups.keys()])

    counts.append(sorted['disltype'].value_counts(sort=False))
    length.append(sorted.groupby('disltype', sort=False)['rnorm'].sum()*bmag*1e6)

    bins = np.arange(0, sorted['disltype'].nunique()+1, 1)
    hist_dict = {'bins': bins, 'weights': sorted['rnorm']*bmag*1e6}

    # hist = sorted['disltype'].hist(ax=ax, xrot=90, rwidth=0.5, align='left', grid=False, alpha=0.5, label=casename.split('_')[1], **hist_dict)
    axs = sorted.hist('angle', by='disltype', bins=15, alpha=0.5, figsize=(12, 9), sharex=True, layout=(3, 3))

    for ax in axs.flatten():
        ax.set_xlabel('Angle (degree)')
        ax.set_ylabel('Counts')
        ax.set_xlim(0, 90)
        ax.set_xticks(np.arange(0, 181, 30)//2)
        ax.tick_params(direction='in')
    figname = os.path.join('configs', 'config_%s'%casename, 'sliptype.png')
    plt.savefig(figname, dpi=300)
    plt.close()

    ax = sorted['angle'].plot(kind='hist', bins=15, alpha=0.5, figsize=(8, 6))
    ax.set_xlabel('Angle (degree)', fontsize=fs)
    ax.set_ylabel('Counts', fontsize=fs)
    ax.set_xlim(0, 90)
    ax.set_xticks(np.arange(0, 181, 30)//2)
    ax.tick_params(direction='in', labelsize=fs-2)
    figname = os.path.join('configs', 'config_%s'%casename, 'angles.png')
    plt.savefig(figname, dpi=300)
    plt.close()

# ax.legend()
# ax.set_xlabel('Slip systems', fontsize=fs)
# ax.set_ylabel(r'Dislocation length (${\rm\mu}$m)', fontsize=fs)
# fig.tight_layout()
# plt.show()

# %%
# Plot the dislocation length

df_counts = pd.concat(counts, axis=1, keys=[int(casename.split('_')[1][2:]) for casename in casenames[:]])
df_counts = df_counts.transpose().sort_index(axis=0)

df_length = pd.concat(length, axis=1, keys=[int(casename.split('_')[1][2:]) for casename in casenames[:]])
df_length = df_length.transpose().sort_index(axis=0)

ax = df_length.plot(kind='line', marker='o', figsize=(8, 6))
ax.set_yscale('log'); ax.set_yticklabels(['%d'%v for v in ax.get_yticks()])
ax.set_xlabel('MD steps', fontsize=fs)
ax.set_ylabel(r'Dislocation length (${\rm\mu}$m)', fontsize=fs)
ax.set_xticks(df_length.index)
ax.tick_params(direction='in', labelsize=fs-2)
ax.legend(title='Dislocation type')
figname = os.path.join('configs', 'dislocation_length.png')
plt.savefig(figname, dpi=300)
plt.close()

# %%
# Save the subset of dislocation network

# ind = 2

# casename = casenames[ind]
for ind, casename in enumerate(casenames[:]):
    print(casename)
    # save_ca_file = 'config_%s.ca'%casename
    # rn, links = load_disl_network(casename, verbose=True, save_ca_file=save_ca_file, reduced=False)

    for i, slip_type in enumerate(disltypes.keys()):
        # save_ca_file = 'sliptype_%s.ca'%(slip_type.replace(' ', '_').replace(r'<', '[').replace(r'>', ']').replace(r'{', '(').replace(r'}', ')'))
        save_ca_file = 'sliptype_%d.ca'%(i)
        save_ca_segs = os.path.join('configs', 'config_%s'%casename, 'sliptype_%d.txt'%i)
        print(save_ca_file, slip_type)
        select_seg = group_list[ind][slip_type].index.to_numpy()
        np.savetxt(save_ca_segs, select_seg, fmt='%d', header=slip_type)
        rn, links = load_disl_network(casename, verbose=False, select_seg=select_seg, save_ca_file=save_ca_file, reduced=False)

# %%