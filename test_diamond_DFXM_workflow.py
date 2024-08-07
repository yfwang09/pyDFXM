#%%-----------------------------------------------------------------
# DFXM forward calculation for diamond DDD configurations
#
# Developer: Yifan Wang, yfwang09@stanford.edu
# Date: 2024/01/15
# Last update: 2024/05/27
#---------------------------------------------------------------

import os, time
import numpy as np
import matplotlib.pyplot as plt
import dispgrad_func as dgf
import forward_model as fwd
import visualize_helper as vis
import disl_io_helper as dio

import argparse

parser = argparse.ArgumentParser(description='DFXM forward calculation for diamond DDD configurations')
parser.add_argument('--casename', '-n', type=str, default='diamond_MD20000_189x100x100', help='The name of the DDD configuration')
parser.add_argument('--scale_cell', '-sc', type=float, default=0.5, help='Scale the cell side by this scale (default = 1)')
parser.add_argument('--poisson', '-nu', type=float, default=0.200, help="Poisson's ratio")
parser.add_argument('--bmag', '-b', type=float, default=2.522e-10, help="Burger's magnitude (m)")
parser.add_argument('--diffraction_plane', '-hkl', type=str, default='400', help='Diffraction plane of diamond (004 or 111)')
parser.add_argument('--rocking', '-phi', type=float, default=0, help='Rocking angle (deg) for the DFXM')
parser.add_argument('--rolling', '-chi', type=float, default=0, help='Rolling angle (deg) for the DFXM')
parser.add_argument('--shift', '-sh', type=float, default=[0, 0, 0], nargs='+', help='Shift of the observation points (um)')
parser.add_argument('--cutoff', '-c', type=float, default=0.51, help='Cutoff distance for the observation region (in scaled coordinates)')
parser.add_argument('--slip', '-s', type=int, default=None, help='slip system')
args, unknown = parser.parse_known_args()

casename   = args.casename
scale_cell = args.scale_cell
phi, chi = args.rocking, args.rolling
shift = args.shift
cutoff = args.cutoff

# For debugging purposes
# casename = 'diamond_MD200000_114x100x107'
# scale_cell = 0.5
# args.diffraction_plane = '400'
# args.slip = 6

slipstr = ''
if args.slip is not None:
    config_dir = os.path.join('configs', 'config_%s'%casename)
    slipstr = '_slip%d'%args.slip
    sliptype_file = os.path.join(config_dir, 'sliptype_%d.txt'%args.slip)
    with open(sliptype_file, 'r') as f:
        sliptype = '_'.join(f.readline().split()[1:])
    select_seg_slip = np.loadtxt(sliptype_file, dtype=int)

#%%
# Define the elasticity properties of diamond
# and the input dictionary for the dislocation network

input_dict = dgf.default_dispgrad_dict('disl_network')

input_dict['nu'] = NU = 0.200       # Poisson's ratio
input_dict['b'] = bmag = 2.522e-10  # Burger's magnitude (m)

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

#%%
# Define the geometry

rn, links = load_disl_network(casename, scale_cell=scale_cell)

diffraction_plane = args.diffraction_plane
if diffraction_plane == '004':
    # Diffraction plane of diamond (004)
    two_theta = 48.16                   # 2theta for diamond-(004) (deg)
    hkl = [0, 0, 1]                     # hkl for diamond-(004) plane
    x_c = [1, 0, 0]                     # x_c for diamond-(004) plane
    y_c = [0, 1, 0]                     # y_c for diamond-(004) plane
elif diffraction_plane == '400':
    two_theta = 48.16                   # 2theta for diamond-(004) (deg)
    hkl = [1, 0, 0]                     # hkl for diamond-(004) plane
    x_c = [0, 1, 0]                     # x_c for diamond-(004) plane
    y_c = [0, 0, 1]                     # y_c for diamond-(004) plane
elif diffraction_plane == '040':
    two_theta = 48.16                   # 2theta for diamond-(004) (deg)
    hkl = [0, 1, 0]                     # hkl for diamond-(004) plane
    x_c = [0, 0, 1]                     # x_c for diamond-(004) plane
    y_c = [1, 0, 0]                     # y_c for diamond-(004) plane
elif diffraction_plane == '111':
    # Diffraction plane of diamond (111)
    two_theta = 20.06                   # 2theta for diamond-(111) (deg)
    hkl = [1, 1, 1]                     # hkl for diamond-(111) plane
    x_c = [1, 1, -2]                    # x_c for diamond-(111) plane
    y_c = [-1, 1, 0]                    # y_c for diamond-(111) plane
elif diffraction_plane == '11-1':
    # Diffraction plane of diamond (111)
    two_theta = 20.06                   # 2theta for diamond-(111) (deg)
    hkl = [1, 1,-1]                     # hkl for diamond-(111) plane
    x_c = [2,-1, 1]                     # x_c for diamond-(111) plane
    y_c = [0, 1, 1]                     # y_c for diamond-(111) plane
elif diffraction_plane == '1-11':
    # Diffraction plane of diamond (111)
    two_theta = 20.06                   # 2theta for diamond-(111) (deg)
    hkl = [1,-1, 1]                     # hkl for diamond-(111) plane
    x_c = [2, 1, -1]                    # x_c for diamond-(111) plane
    y_c = [0, 1, 1]                     # y_c for diamond-(111) plane
elif diffraction_plane == '-111':
    # Diffraction plane of diamond (111)
    two_theta = 20.06                   # 2theta for diamond-(111) (deg)
    hkl = [-1, 1, 1]                    # hkl for diamond-(111) plane
    x_c = [ 2, 1, 1]                    # x_c for diamond-(111) plane
    y_c = [0, 1, -1]                    # y_c for diamond-(111) plane
else:
    raise ValueError('Unknown diffraction plane: %s'%diffraction_plane)

casename_noslip_scaled = casename + '_scale%d'%(1/scale_cell)
casename_scaled = casename + slipstr + '_scale%d'%(1/scale_cell)

phi = chi = 0
shift = [0, 0, 0]

casename_noslip_scaled_hkl = casename_noslip_scaled + '_shift-%.2f-%.2f-%.2f'%tuple(shift) + '_hkl%d%d%d'%tuple(hkl)
casename_scaled_hkl = casename_scaled + '_shift-%.2f-%.2f-%.2f'%tuple(shift) + '_hkl%d%d%d'%tuple(hkl)

forward_dict = fwd.default_forward_dict()
forward_dict['two_theta'] = two_theta
forward_dict['hkl'] = hkl
forward_dict['x_c'] = x_c
forward_dict['y_c'] = y_c
forward_dict['phi'] = phi
forward_dict['chi'] = chi

print(forward_dict)

#%%

# Set up the pre-calculated resolution function
datapath = 'data'
os.makedirs(datapath, exist_ok=True)
Fg_path = os.path.join(datapath, 'Fg_%s_seg'%casename)
os.makedirs(Fg_path, exist_ok=True)
im_path = os.path.join(datapath, 'im_%s'%casename)
if args.slip is not None:
    im_path = os.path.join(im_path, 'slip%d'%args.slip)
os.makedirs(im_path, exist_ok=True)
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

# load empty network to calculate the observation points
_ = load_disl_network(casename, scale_cell=scale_cell, select_seg=[])
saved_Fg_file = os.path.join(datapath, 'Fg_%s_DFXM_robs.npz'%casename_noslip_scaled_hkl)
print('saved observation points at %s'%saved_Fg_file)
Fg_func = lambda x, y, z: disl.Fg(x, y, z, filename=saved_Fg_file)
if not os.path.exists(saved_Fg_file):
    im, ql, rulers = model.forward(Fg_func, timeit=True)
    Fg = np.load(saved_Fg_file)['Fg']*0
    r_obs = np.load(saved_Fg_file)['r_obs'] + np.multiply(shift, 1e-6) # in the unit of m
    np.savez_compressed(saved_Fg_file, Fg=Fg, r_obs=r_obs)

# Visualize the observation points (in unit of um)
Lx, Ly, Lz = tuple(np.diag(disl.cell))
lbx, ubx = -Lx/2*bmag, Lx/2*bmag         # in unit of m
lby, uby = -Ly/2*bmag, Ly/2*bmag         # in unit of m
lbz, ubz = -Lz/2*bmag, Lz/2*bmag         # in unit of m
extent = np.multiply(1e6, [lbx, ubx, lby, uby, lbz, ubz]) # in the unit of um
fig, ax = vis.visualize_disl_network(disl.d, disl.rn, disl.links, extent=extent, unit='um', show=False) # draw the empty box

Nobs = 2
NNxyz = np.multiply(model.d['Npixels'], Nobs)
# NNxyz = (100, 80, 90)
NNxyz[1], NNxyz[2] = NNxyz[2], NNxyz[1]
NNxyz = tuple(NNxyz)
print(NNxyz)
r_obs = np.load(saved_Fg_file)['r_obs'] * 1e6 # in the unit of um
r_obs_cell = np.swapaxes(np.reshape(r_obs, NNxyz + (3, ), order='F'), 1, 2)

ax.plot(r_obs_cell[:, 0, 0, 0], r_obs_cell[:, 0, 0, 1], r_obs_cell[:, 0, 0, 2], '-C0')
ax.plot(r_obs_cell[0, :, 0, 0], r_obs_cell[0, :, 0, 1], r_obs_cell[0, :, 0, 2], '-C1')
ax.plot(r_obs_cell[0, 0, :, 0], r_obs_cell[0, 0, :, 1], r_obs_cell[0, 0, :, 2], '-C2')
ax.plot(r_obs_cell[:, 0, -1, 0], r_obs_cell[:, 0, -1, 1], r_obs_cell[:, 0, -1, 2], '-k')
ax.plot(r_obs_cell[0, :, -1, 0], r_obs_cell[0, :, -1, 1], r_obs_cell[0, :, -1, 2], '-k')
ax.plot(r_obs_cell[0, -1, :, 0], r_obs_cell[0, -1, :, 1], r_obs_cell[0, -1, :, 2], '-k')
ax.plot(r_obs_cell[:, -1, 0, 0], r_obs_cell[:, -1, 0, 1], r_obs_cell[:, -1, 0, 2], '-k')
ax.plot(r_obs_cell[-1, :, 0, 0], r_obs_cell[-1, :, 0, 1], r_obs_cell[-1, :, 0, 2], '-k')
ax.plot(r_obs_cell[-1, 0, :, 0], r_obs_cell[-1, 0, :, 1], r_obs_cell[-1, 0, :, 2], '-k')
ax.plot(r_obs_cell[:, -1, -1, 0], r_obs_cell[:, -1, -1, 1], r_obs_cell[:, -1, -1, 2], '-k')
ax.plot(r_obs_cell[-1, :, -1, 0], r_obs_cell[-1, :, -1, 1], r_obs_cell[-1, :, -1, 2], '-k')
ax.plot(r_obs_cell[-1, -1, :, 0], r_obs_cell[-1, -1, :, 1], r_obs_cell[-1, -1, :, 2], '-k')


nskip = 10
ax.plot(r_obs[::nskip, 0], r_obs[::nskip, 1], r_obs[::nskip, 2],  'C0.', markersize=0.01)
# ax.view_init(azim=90, elev=20)
plt.show()

# %%-------------------------------------------------------
# FILTER THE DISLOCATION SEGMENTS THAT ARE OUTSIDE THE OBSERVATION REGION
#---------------------------------------------------------

r_obs = np.load(saved_Fg_file)['r_obs'] # in the unit of m
r_obs_cell = np.swapaxes(np.reshape(r_obs, NNxyz + (3, ), order='F'), 1, 2)
obs_cell = np.transpose([
    r_obs_cell[-1, 0, 0, :] - r_obs_cell[0, 0, 0, :], 
    r_obs_cell[0, -1, 0, :] - r_obs_cell[0, 0, 0, :],
    r_obs_cell[0, 0, -1, :] - r_obs_cell[0, 0, 0, :]
])                      # in unit of m
print(obs_cell*1e6, 'um')
rn, links = load_disl_network(casename, scale_cell=scale_cell)
nsegs = disl.links.shape[0]
select_seg_inside = []
for ilink in range(nsegs):
    link = disl.links[ilink]
    end1 = disl.rn[int(link[0])]*bmag - np.multiply(shift, 1e-6)
    end2 = disl.rn[int(link[1])]*bmag - np.multiply(shift, 1e-6)
    s1 = np.linalg.inv(obs_cell).dot(end1)
    s2 = np.linalg.inv(obs_cell).dot(end2)
    if np.all(np.abs(s1) < 0.51) or np.all(np.abs(s2) < 0.51):
        select_seg_inside.append(ilink)
print('# of segments inside the observation region:', len(select_seg_inside))

# if os.path.exists(saved_Fg_file):
#     os.remove(saved_Fg_file)

# %% --------------------------------------------------------
# SAVE THE DISLOCATION NETWORK INSIDE THE OBSERVATION REGION
# -----------------------------------------------------------

if args.slip is not None:
    select_seg = np.intersect1d(select_seg_slip, select_seg_inside).tolist()
else:
    select_seg = select_seg_inside
print(len(select_seg))

config_ca_inside_file = os.path.join('config_%s_inside.ca'%casename_scaled_hkl)
rn, links = load_disl_network(casename, scale_cell=scale_cell, select_seg=select_seg, save_ca_file=config_ca_inside_file)

# %% --------------------------------------------------------
# CALCULATE THE DFXM IMAGE
# -----------------------------------------------------------

print('#'*20 + ' Calculate and visualize the image')
saved_Fg_file = os.path.join(datapath, 'Fg_%s_DFXM.npz'%casename_scaled_hkl)
print('saved dispgrad at %s'%saved_Fg_file)

if not os.path.exists(saved_Fg_file):
    Fg_saved = np.zeros(r_obs.shape + (3, ))
    for iseg, seg in enumerate(select_seg):
        saved_Fg_seg_file = os.path.join(Fg_path, 'Fg_%s'%(casename_noslip_scaled_hkl, )) + '_iseg%d_DFXM.npz'%(seg, )
        if os.path.exists(saved_Fg_seg_file):
            print('load %s'%saved_Fg_seg_file)
            Fg_seg = np.load(saved_Fg_seg_file)['Fg']
        else:
            # print('zero %s'%saved_Fg_seg_file)
            Fg_seg = 0
        Fg_saved += Fg_seg
    np.savez_compressed(saved_Fg_file, Fg=Fg_saved)

Fg_func = lambda x, y, z: disl.Fg(x, y, z, filename=saved_Fg_file)
im, ql, rulers = model.forward(Fg_func, timeit=True)

# Visualize the simulated image
figax = vis.visualize_im_qi(forward_dict, im, None, rulers) #, vlim_im=[0, 200])

# Visualize the reciprocal space wave vector ql
# figax = vis.visualize_im_qi(forward_dict, None, ql, rulers, vlim_qi=[-1e-4, 1e-4])

# %%
# save r_obs into xyz file

r_obs_xyz_file = os.path.join(datapath, 'r_obs_%s.xyz.gz'%casename_noslip_scaled_hkl)
im_Nobs = np.repeat(np.repeat(im, Nobs, axis=0), Nobs, axis=1)[:,:,np.newaxis]
im_obs = np.tile(im_Nobs, (1, 1, model.d['Npixels'][2]*Nobs)).reshape((-1, 1))
r_obs = r_obs_cell.reshape((-1, 3))
lat_str = '"%s %s %s %s %s %s %s %s %s"'%tuple(obs_cell.T.flatten()/scale_cell/1e-10)
org_str = '"%s %s %s"'%tuple(r_obs_cell[0,0,0,:]/scale_cell/1e-10)
if not os.path.exists(r_obs_xyz_file):
    dio.write_xyz(r_obs_xyz_file, r_obs, props=im_obs, scale=(1/scale_cell)/1e-10, parameters={'Lattice': lat_str, 'Origin': org_str, 'Properties': 'pos:R:3:Intensity:R:1', 'pbc': '"F F F"'})

# %%
# Calculating the rocking curve

phi_values = np.arange(-0.001, 0.00101, 0.0001).round(4)
chi = 0
Imin = np.empty_like(phi_values)
Imax = np.empty_like(phi_values)
Iavg = np.empty_like(phi_values)

for iphi, phi in enumerate(phi_values):
    casename_scaled_phi_chi_hkl = casename_scaled + '_phi%.5f'%phi + '_chi%.5f'%chi + '_shift-%.2f-%.2f-%.2f'%tuple(shift) + '_hkl%d%d%d'%tuple(hkl)
    print('#'*20 + ' Calculate and visualize the image')
    saved_Fg_file = os.path.join(datapath, 'Fg_%s_DFXM.npz'%casename_scaled_hkl)
    print('saved dispgrad at %s'%casename_scaled_phi_chi_hkl)

    model.d['phi'] = phi
    model.d['chi'] = chi
    Fg_func = lambda x, y, z: disl.Fg(x, y, z, filename=saved_Fg_file)
    saved_im_file = os.path.join(im_path, 'im_%s_DFXM'%casename_scaled_phi_chi_hkl)
    if os.path.exists(saved_im_file+'.npz'):
        rawdata = np.load(saved_im_file+'.npz', allow_pickle=True)
        im, ql, rulers = rawdata['im'], rawdata['ql'], rawdata['rulers']
        print('load im_file', saved_im_file+'.npz')
    else:
        im, ql, rulers = model.forward(Fg_func, timeit=True)
        print('Calculate im')
    if not os.path.exists(saved_im_file+'.png'):
        figax = vis.visualize_im_qi(forward_dict, im, None, rulers)
        print('save', saved_im_file+'.png')
        figax[0].savefig(saved_im_file+'.png', dpi=300, transparent=True)
        plt.close()
    if not os.path.exists(saved_im_file+'.xyz.gz'):
        im_Nobs = np.repeat(np.repeat(im, Nobs, axis=0), Nobs, axis=1)[:,:,np.newaxis]
        # im_obs = np.tile(im_Nobs, (1, 1, model.d['Npixels'][2]*Nobs)).reshape((-1, 1))
        # r_obs = r_obs_cell.reshape((-1, 3))
        Nz = model.d['Npixels'][2]
        r_obs = r_obs_cell[:, :, Nz-1:Nz+1].reshape((-1, 3))
        im_obs = np.tile(im_Nobs, (1, 1, 2)).reshape((-1, 1))
        print('save', saved_im_file+'.xyz.gz')
        dio.write_xyz(saved_im_file+'.xyz.gz', r_obs, props=im_obs, scale=(1/scale_cell)/1e-10, parameters={'Lattice': lat_str, 'Origin': org_str, 'Properties': 'pos:R:3:Intensity:R:1', 'pbc': '"F F F"'})
    
    Imin[iphi], Imax[iphi], Iavg[iphi] = im.min(), im.max(), im.mean()

saved_rocking_curve = os.path.join(im_path, 'im_%s'%(casename_scaled)+'_hkl%d%d%d'%tuple(hkl)+'_rocking_DFXM.png')

fig, ax = plt.subplots()
# ax.plot(phi_values, Imax, label=r'$I_{\rm max}$')
# ax.plot(phi_values, Imin, label=r'$I_{\rm min}$')
vphi = np.multiply(phi_values, 1000).round(4)
ax.fill_between(vphi, Imin, Imax, alpha=0.5, color='C0', label=r'$[I_{\rm min}, I_{\rm max}]$')
ax.plot(vphi, Iavg, 'C0', label=r'$I_{\rm avg}$')
ax.legend()
# ax.set_xlabel(r'Rocking $\phi$ (rad)')
ax.set_ylabel('Intensity (a.u.)')

# xticks = ax.get_xticks()
# ax.set_xticklabels(np.round(np.multiply(1000, xticks), 4).tolist())
ax.set_xlabel(r'Rocking $\phi$ (mrad)')

if not os.path.exists(saved_rocking_curve):
    fig.savefig(saved_rocking_curve, dpi=300, transparent=True)
plt.show()


# %%
# Calculating the rolling curve

phi_values = np.arange(-0.002, 0.00201, 0.0001)#.round(4)
chi = 0.0
Imin = np.empty_like(phi_values)
Imax = np.empty_like(phi_values)
Iavg = np.empty_like(phi_values)

for iphi, phi in enumerate(phi_values):
    if np.isclose(phi, 0.0):
        phi = 0
    casename_scaled_phi_chi_hkl = casename_scaled + '_phi%.5f'%chi + '_chi%.5f'%phi + '_shift-%.2f-%.2f-%.2f'%tuple(shift) + '_hkl%d%d%d'%tuple(hkl)
    print('#'*20 + ' Calculate and visualize the image')
    saved_Fg_file = os.path.join(datapath, 'Fg_%s_DFXM.npz'%casename_scaled_hkl)
    print('saved dispgrad at %s'%casename_scaled_phi_chi_hkl)

    model.d['phi'] = chi
    model.d['chi'] = phi
    Fg_func = lambda x, y, z: disl.Fg(x, y, z, filename=saved_Fg_file)
    saved_im_file = os.path.join(im_path, 'im_%s_DFXM'%casename_scaled_phi_chi_hkl)
    if os.path.exists(saved_im_file+'.npz'):
        rawdata = np.load(saved_im_file+'.npz', allow_pickle=True)
        im, ql, rulers = rawdata['im'], rawdata['ql'], rawdata['rulers']
        print('load im_file', saved_im_file+'.npz')
    else:
        im, ql, rulers = model.forward(Fg_func, timeit=True)
        print('Calculate im')
    if not os.path.exists(saved_im_file+'.png'):
        figax = vis.visualize_im_qi(forward_dict, im, None, rulers)
        print('save', saved_im_file+'.png')
        figax[0].savefig(saved_im_file+'.png', dpi=300, transparent=True)
        plt.close()
    if not os.path.exists(saved_im_file+'.xyz.gz'):
        im_Nobs = np.repeat(np.repeat(im, Nobs, axis=0), Nobs, axis=1)[:,:,np.newaxis]
        # im_obs = np.tile(im_Nobs, (1, 1, model.d['Npixels'][2]*Nobs)).reshape((-1, 1))
        # r_obs = r_obs_cell.reshape((-1, 3))
        Nz = model.d['Npixels'][2]
        r_obs = r_obs_cell[:, :, Nz-1:Nz+1].reshape((-1, 3))
        im_obs = np.tile(im_Nobs, (1, 1, 2)).reshape((-1, 1))
        print('save', saved_im_file+'.xyz.gz')
        dio.write_xyz(saved_im_file+'.xyz.gz', r_obs, props=im_obs, scale=(1/scale_cell)/1e-10, parameters={'Lattice': lat_str, 'Origin': org_str, 'Properties': 'pos:R:3:Intensity:R:1', 'pbc': '"F F F"'})

    Imin[iphi], Imax[iphi], Iavg[iphi] = im.min(), im.max(), im.mean()

saved_rocking_curve = os.path.join(im_path, 'im_%s'%(casename_scaled)+'_hkl%d%d%d'%tuple(hkl)+'_rocking_DFXM.png')

fig, ax = plt.subplots()
vphi = np.multiply(phi_values, 1000).round(4)
ax.fill_between(vphi, Imin, Imax, alpha=0.5, color='C0', label=r'$[I_{\rm min}, I_{\rm max}]$')
ax.plot(vphi, Iavg, 'C0', label=r'$I_{\rm avg}$')
ax.legend()
# ax.set_xlabel(r'Rolling $\chi$ (rad)')
ax.set_ylabel('Intensity (a.u.)')

# xticks = ax.get_xticks()
# ax.set_xticklabels(np.round(np.multiply(1000, xticks), 4).tolist())
ax.set_xlabel(r'Rolling $\chi$ (mrad)')

if not os.path.exists(saved_rocking_curve):
    fig.savefig(saved_rocking_curve, dpi=300, transparent=True)
plt.show()

# %%
# Calculating the mosaic space

phi_values = np.arange(-0.001, 0.00101, 0.0001).round(4)
chi_values = np.arange(-0.002, 0.00201, 0.0001).round(4)

saved_mosaic_space = os.path.join(im_path, 'im_%s'%(casename_scaled)+'_hkl%d%d%d'%tuple(hkl)+'_mosaic_data.npz')
# np.savez_compressed(saved_mosaic_space, PHI=PHI, CHI=CHI, Imin=Imin, Imax=Imax, Iavg=Iavg)
if os.path.exists(saved_mosaic_space):
    saved_data = np.load(saved_mosaic_space)
    PHI = saved_data['PHI']
    CHI = saved_data['CHI']
    Imin = saved_data['Imin']
    Imax = saved_data['Imax']
    Iavg = saved_data['Iavg']

else:
    PHI, CHI = np.meshgrid(phi_values, chi_values)
    Imin = np.empty_like(PHI)
    Imax = np.empty_like(PHI)
    Iavg = np.empty_like(PHI)

    for iphi, phi in np.ndenumerate(PHI):
        chi = CHI[iphi]
        casename_scaled_phi_chi_hkl = casename_scaled + '_phi%.5f'%phi + '_chi%.5f'%chi + '_shift-%.2f-%.2f-%.2f'%tuple(shift) + '_hkl%d%d%d'%tuple(hkl)
        print('#'*20 + ' Calculate and visualize the image')
        saved_Fg_file = os.path.join(datapath, 'Fg_%s_DFXM.npz'%casename_scaled_hkl)
        print('saved dispgrad at %s'%casename_scaled_phi_chi_hkl)

        model.d['phi'] = phi
        model.d['chi'] = chi
        Fg_func = lambda x, y, z: disl.Fg(x, y, z, filename=saved_Fg_file)
        saved_im_file = os.path.join(im_path, 'im_%s_DFXM'%casename_scaled_phi_chi_hkl)
        if os.path.exists(saved_im_file+'.npz'):
            rawdata = np.load(saved_im_file+'.npz', allow_pickle=True)
            im, ql, rulers = rawdata['im'], rawdata['ql'], rawdata['rulers']
            print('load im_file', saved_im_file+'.npz')
        else:
            im, ql, rulers = model.forward(Fg_func, timeit=True)
            # rulers is a tuple of ((Nx,), (Ny,), (Nz,))
            # should be saved separately (todo)
            # VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
            np.savez_compressed(saved_im_file, im=im, ql=ql, rulers=rulers)
            print('Calculate and save im_file', saved_im_file+'.npz')
        if not os.path.exists(saved_im_file+'.png'):
            figax = vis.visualize_im_qi(forward_dict, im, None, rulers)
            print('save', saved_im_file+'.png')
            figax[0].savefig(saved_im_file+'.png', dpi=300, transparent=True)
            plt.close()
        if not os.path.exists(saved_im_file+'.xyz.gz'):
            im_Nobs = np.repeat(np.repeat(im, Nobs, axis=0), Nobs, axis=1)[:,:,np.newaxis]
            # im_obs = np.tile(im_Nobs, (1, 1, model.d['Npixels'][2]*Nobs)).reshape((-1, 1))
            # r_obs = r_obs_cell.reshape((-1, 3))
            Nz = model.d['Npixels'][2]
            r_obs = r_obs_cell[:, :, Nz-1:Nz+1].reshape((-1, 3))
            im_obs = np.tile(im_Nobs, (1, 1, 2)).reshape((-1, 1))
            print('save', saved_im_file+'.xyz.gz')
            dio.write_xyz(saved_im_file+'.xyz.gz', r_obs, props=im_obs, scale=(1/scale_cell)/1e-10, parameters={'Lattice': lat_str, 'Origin': org_str, 'Properties': 'pos:R:3:Intensity:R:1', 'pbc': '"F F F"'})

        Imin[iphi], Imax[iphi], Iavg[iphi] = im.min(), im.max(), im.mean()
    np.savez_compressed(saved_mosaic_space, PHI=PHI, CHI=CHI, Imin=Imin, Imax=Imax, Iavg=Iavg)

# %%
# Visualize the mosaic space

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(PHI*1000, CHI*1000, Iavg, cmap='viridis')
ax.set_xlabel(r'$\phi\times10^3$ (rad)')
ax.set_ylabel(r'$\chi\times10^3$ (rad)')
ax.set_zlabel(r'$I_{\rm avg}$')
lims = (np.min(np.stack([PHI, CHI]))*1000, np.max(np.stack([PHI, CHI]))*1000)
ax.set_xlim(*lims)
ax.set_ylim(*lims)
saved_mosaic_space = os.path.join(im_path, 'im_%s'%(casename_scaled)+'_hkl%d%d%d'%tuple(hkl)+'_mosaic_3D.png')
fig.savefig(saved_mosaic_space, dpi=300, transparent=True)
plt.show()

fig, ax = plt.subplots()
ax.imshow(Iavg, cmap='viridis', extent=np.multiply(1000, [phi_values.min(), phi_values.max(), chi_values.min(), chi_values.max()]))
ax.axis('equal')
ax.set_xlabel(r'$\phi\times10^3$ (rad)')
ax.set_ylabel(r'$\chi\times10^3$ (rad)')
saved_mosaic_space = os.path.join(im_path, 'im_%s'%(casename_scaled)+'_hkl%d%d%d'%tuple(hkl)+'_mosaic_2D.png')
fig.savefig(saved_mosaic_space, dpi=300, transparent=True)
plt.show()

# %%
