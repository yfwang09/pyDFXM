#%%-----------------------------------------------------------------
# DFXM forward calculation for a triangular dislocation loop
#
# The test case for the displacement gradient field is in test_triangular_loop_dispgrad.py
#
# Note that rn defined in the disl object is in the unit of Burger's vector,
# while the r in the DFXM calculation is in the unit of meter.
# The bmag is only used in the Fg function for DFXM calculation 
# (in test_triangular_loop_DFXM.py)
#
# Developer: Yifan Wang, yfwang09@stanford.edu
# Date: 2023/12/27
# updated: 2024/07/10
#---------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
import dispgrad_func as dgf
import forward_model as fwd
import visualize_helper as vis

# material = 'Al'; diffraction_plane = '002'
material = 'diamond'; diffraction_plane = '004'

#%%-------------------------------------------------------
# INITIALIZATION
#---------------------------------------------------------

# Elasticity parameters (Aluminum)
input_dict = dgf.default_dispgrad_dict('disl_network')

if material == 'Al':
    input_dict['nu'] = NU = 0.324       # Poisson's ratio
    input_dict['b'] = bmag = 2.86e-10   # Burger's magnitude (m)
elif material == 'diamond':
    input_dict['nu'] = NU = 0.200       # Poisson's ratio
    input_dict['b'] = bmag = 2.522e-10  # Burger's magnitude (m)

# Initialize the triangular loop 
L = 50000        # in the unit of b
rn = np.subtract([[1*L/4, 3*L/4, L],
                  [3*L/4, L, 1*L/4],
                  [L, 1*L/4, 3*L/4]]
                  , [2*L/3, 2*L/3, 2*L/3])/6
print(rn)
# Normalized Burger's vector
b = np.array([1, -1, 0])
b = b/np.linalg.norm(b)
# Normalized slip plane normal
n = np.array([1, 1, 1])
n = n/np.linalg.norm(n)
# Connectivity
links = np.transpose([[0, 1, 2], [1, 2, 0]])
links = np.hstack([links, np.tile(b, (3, 1)), np.tile(n, (3, 1))])

# Initialize the dislocation network object
input_dict['rn'] = rn
input_dict['links'] = links
disl = dgf.disl_network(input_dict)

#%%-------------------------------------------------------
# CALCULATE THE DISPLACEMENT GRADIENT
#---------------------------------------------------------

forward_dict = fwd.default_forward_dict()

# if you want to change the forward model setup
# forward_dict['hkl'] = [1, 1, 1]
# forward_dict['x_c'] = [1, -1, 0]
# forward_dict['y_c'] = [-1, 1, 0]
# forward_dict['phi'] = np.deg2rad(0.001)

if material == 'Al':
    if diffraction_plane == '002':
        # Diffraction plane of Al (002)
        two_theta = 20.73                   # 2theta for Al-(002) (deg)
        hkl = [0, 0, 1]                     # hkl for Al-(001) plane
        x_c = [1, 0, 0]                     # x_c for Al-(001) plane
        y_c = [0, 1, 0]                     # y_c for Al-(001) plane
    elif diffraction_plane == '111':
        # Diffraction plane of Al (111)
        raise NotImplementedError('The Al (111) plane is not implemented yet.')
elif material == 'diamond':
    if diffraction_plane == '004':
        # Diffraction plane of diamond (004)
        two_theta = 48.16                   # 2theta for diamond-(004) (deg)
        hkl = [0, 0, 1]                     # hkl for diamond-(004) plane
        x_c = [1, 0, 0]                     # x_c for diamond-(004) plane
        y_c = [0, 1, 0]                     # y_c for diamond-(004) plane
    elif diffraction_plane == '111':
        # Diffraction plane of diamond (111)
        two_theta = 20.06                   # 2theta for diamond-(111) (deg)
        hkl = [1, 1, 1]                     # hkl for diamond-(111) plane
        x_c = [1, 1, -2]                    # x_c for diamond-(111) plane
        y_c = [-1, 1, 0]                    # y_c for diamond-(111) plane

forward_dict['two_theta'] = two_theta
forward_dict['hkl'] = hkl
forward_dict['x_c'] = x_c
forward_dict['y_c'] = y_c

print(forward_dict)

# Set up the pre-calculated resolution function
datapath = 'data'
os.makedirs(datapath, exist_ok=True)
saved_res_fn = os.path.join(datapath, 'Res_qi_%s_%s.npz'%(material, diffraction_plane))
print('saved resolution function at %s'%saved_res_fn)

model = fwd.DFXM_forward(forward_dict, load_res_fn=saved_res_fn)
Ug = model.Ug
print('Ug')
print(Ug)

# or you can change the parameters here
# model.d['phi'] = np.deg2rad(0.001)

lb, ub = -L/2*bmag, L/2*bmag            # in unit of m
Ngrid = 50
xs = np.linspace(lb, ub, Ngrid)         # (Ngrid, )
ys = np.linspace(lb, ub, Ngrid)         # (Ngrid, )
zs = np.linspace(lb, ub, Ngrid)         # (Ngrid, )

XX, YY, ZZ = np.meshgrid(xs, ys, zs)    # (Ngrid, Ngrid, Ngrid)
Rs = np.stack([XX,YY,ZZ], -1)           # (Ngrid, Ngrid, Ngird, 3)
print('Grid size in the sample coordinates:', Rs.shape)

print('Convert Rs into the grain coordinates (Miller indices)')
Rg = np.einsum('ij,...j->...i', Ug, Rs)
Fg = disl.Fg(Rg[..., 0], Rg[..., 1], Rg[..., 2])

print('Visualize the displacement gradient')
i, j = 1, 2
vmin, vmax = -1e-3, 1e-3
fs = 12
extent = np.multiply(1e6, [lb, ub, lb, ub, lb, ub]) # in the unit of um
# figax = vis.plot_3d_slice_z(Fg[:, :, :, i, j], extent=extent, vmin=vmin, vmax=vmax, nslice=5, fs=fs, show=False)
# figax = vis.visualize_disl_network(disl.d, rn, links, unit='um', figax=figax, show=False)
# figax = vis.visualize_disl_network(disl.d, rn, links, unit='um')
# plt.show()

#%%-------------------------------------------------------
# CALCULATE THE DFXM IMAGE
#---------------------------------------------------------

print('#'*20 + ' Calculate and visualize the image')
saved_Fg_file = os.path.join(datapath, 'Fg_triloop_%s_%s.npz'%(material, diffraction_plane))
print('saved displacement gradient at %s'%saved_Fg_file)

# Strong beam condition
Fg_func = lambda x, y, z: disl.Fg(x, y, z, filename=saved_Fg_file)
model.d['phi'] = 0.0
im, ql, rulers = model.forward(Fg_func)
im_max_strong = im.max()

# Visualize the simulated image
im = im/np.max(im)
# figax = vis.visualize_im_qi(forward_dict, im, None, rulers, unit='um', vlim_im=[0, 1], cbar=False)

# Load previously calculated observation points
r_obs = np.load(saved_Fg_file)['r_obs']*1e6 # in the unit of um
fig, ax = vis.visualize_disl_network(disl.d, disl.rn, disl.links, extent=extent, unit='um', show=False)
nskip = 223
ax.plot(r_obs[::nskip, 0], r_obs[::nskip, 1], r_obs[::nskip, 2],  'C0.', markersize=0.01)
# plt.show()
print(r_obs.shape)

#%%-------------------------------------------------------
# NEW DEVELOPMENT OF OBSERVATION POINTS
#---------------------------------------------------------

d = model.d
if type(d['Npixels']) is int:
    Nx = Ny = Nz = d['Npixels']
else:
    Nx, Ny, Nz = d['Npixels']
Nsub = 2                # multiply 2 to avoid sampling the 0 point, make the grids symmetric over 0
NNx, NNy, NNz = Nsub*Nx, Nsub*Ny, Nsub*Nz

# INPUT instrumental settings, related to direct space resolution function
psize = d['psize']   # pixel size in units of m, in the object plane
zl_rms = d['zl_rms'] # rms value of Gaussian beam profile, in m, centered at 0
theta_0 = np.deg2rad(d['two_theta']/2) # in rad
v_hkl = d['hkl']
TwoDeltaTheta = d['TwoDeltaTheta']
U = d['Ug']
phi = d['phi']
chi = d['chi']
omega = d['omega']
eta = np.deg2rad(30) # in rad

# Grid size in the imaging coordinates
yi_start = -psize*Ny/2 + psize/(2*Nsub) # start in yi direction, in units of m, centered at 0
yi_step = psize/Nsub
xi_start = -psize*Nx/2 + psize/(2*Nsub) # start in xi direction, in m, for zi=0
xi_step = psize/Nsub
zl_start = -0.5*zl_rms*6 # start in zl direction, in m, for zl=0
zl_step = zl_rms*6/(NNz-1)

# rotation of the imaging coordinates
mu = theta_0
M = np.matrix([[np.cos(mu), 0, np.sin(mu)],
    [0, 1, 0],
    [-np.sin(mu), 0, np.cos(mu)],
]) # clockwise rotation around y-axis
Omega = np.matrix(
    [[np.cos(omega), np.sin(omega), 0],
     [-np.sin(omega), np.cos(omega), 0],
     [0, 0, 1],
]) # counter-clockwise rotation around z-axis
Chi = np.eye(3)
Phi = np.eye(3)
Gamma = M@Omega@Chi@Phi
theta = (theta_0 + TwoDeltaTheta/2)
Theta = np.matrix([[np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
]) # clockwise rotation around y-axis
Eta = np.matrix([[1, 0, 0], 
       [0, np.cos(eta), np.sin(eta)],
       [0, -np.sin(eta), np.cos(eta)]
      ]) # counter-clockwise rotation around x-axis
Delta = Theta@Theta@Eta

yi0 = yi_start + np.arange(NNy)*yi_step
xi0 = xi_start + np.arange(NNx)*xi_step
zi0 = [0, ]
zi1 = [5e-6, ] # 1e-6 is a small number to avoid singularity in the calculation of the displacement gradient
ZI0, YI0, XI0 = np.meshgrid(xi0, yi0, zi0) # (NNy, NNx, 1)
ZI1, YI1, XI1 = np.meshgrid(xi0, yi0, zi1) # (NNy, NNx, 1)
# print('ZI0 shape:', ZI0.shape, YI0.shape, XI0.shape)
# print('ZI1 shape:', ZI1.shape, YI1.shape, XI1.shape)
RI0 = np.stack([XI0, YI0, ZI0], -1) # (NNy, NNx, 1, 3)
RI1 = np.stack([XI1, YI1, ZI1], -1) # (NNy, NNx, 1, 3)
# print('RI0 shape:', RI0.shape)
# print('RI1 shape:', RI1.shape)

# # Eta rotation
# xyz0 = np.einsum('ij,...j->...i', np.transpose(Eta), RI).reshape(-1, 3)*1e6 # (1*NNy*NNx, 3)
# fig, ax = vis.visualize_disl_network(disl.d, disl.rn, disl.links, extent=extent, unit='um', show=False)
# ax.plot(xyz0[:, 0], xyz0[:, 1], xyz0[:, 2], 'C0.', markersize=0.01)
# ax.view_init(elev=0, azim=0)
# ax.set_title('Eta rotation')
# plt.show()
# # Theta rotation
# xyz0 = np.einsum('ij,...j->...i', np.transpose(Theta@Theta), RI).reshape(-1, 3)*1e6 # (1*NNy*NNx, 3)
# fig, ax = vis.visualize_disl_network(disl.d, disl.rn, disl.links, extent=extent, unit='um', show=False)
# ax.plot(xyz0[:, 0], xyz0[:, 1], xyz0[:, 2], 'C0.', markersize=0.01)
# ax.view_init(elev=0, azim=-90)
# ax.set_title('Theta rotation')
# plt.show()
# # Omega rotation
# xyz0 = np.einsum('ij,...j->...i', np.transpose(Omega), RI).reshape(-1, 3)*1e6 # (1*NNy*NNx, 3)
# fig, ax = vis.visualize_disl_network(disl.d, disl.rn, disl.links, extent=extent, unit='um', show=False)
# ax.plot(xyz0[:, 0], xyz0[:, 1], xyz0[:, 2], 'C0.', markersize=0.01)
# ax.view_init(elev=90, azim=0)
# ax.set_title('Omega rotation')
# plt.show()

RL0 = np.einsum('ij,...j->...i', Delta.T, RI0) # (NNy, NNx, 1, 3)
RL1 = np.einsum('ij,...j->...i', Delta.T, RI1) # (NNy, NNx, 1, 3)
print('RL0 shape:', RL0.shape)
print('RL1 shape:', RL1.shape)
xyz0 = RL0.reshape(-1, 3)*1e6 # (1*NNy*NNx, 3)
xyz1 = RL1.reshape(-1, 3)*1e6 # (1*NNy*NNx, 3)
line01 = np.stack([RL0, RL1], axis=0) # (2, NNy, NNx, 1, 3)
print('line01 shape:', line01.shape)

# find the crossing points of line01 at each zl values

zl = zl_start + np.arange(NNz)*zl_step # (NNz, )

tl = (zl[None, None, :] - line01[0, ..., 2]) / (line01[1, ..., 2] - line01[0, ..., 2]) # (NNy, NNx, NNz)
XL = tl*(line01[1, ..., 0] - line01[0, ..., 0]) + line01[0, ..., 0] # (NNy, NNx, NNz)
YL = tl*(line01[1, ..., 1] - line01[0, ..., 1]) + line01[0, ..., 1] # (NNy, NNx, NNz)
ZL = tl*(line01[1, ..., 2] - line01[0, ..., 2]) + line01[0, ..., 2] # (NNy, NNx, NNz)
RL = np.stack([XL, YL, ZL], axis=3) # (NNz, NNy, NNx, 3)
print('RL shape:', RL.shape)

visRL = RL.reshape(NNz, -1, 3)*1e6 # in the unit of um

# visualize the new observation points
fig, ax = vis.visualize_disl_network(disl.d, disl.rn, disl.links, extent=extent, unit='um', show=False)

ax.plot(xyz0[:, 0], xyz0[:, 1], xyz0[:, 2], 'C0.', markersize=0.01)
ax.plot(xyz1[:, 0], xyz1[:, 1], xyz1[:, 2], 'C1.', markersize=0.01)
for i in range(0, visRL.shape[1], nskip):
    ax.plot(visRL[:, i, 0], visRL[:, i, 1], visRL[:, i, 2], 'C4-', linewidth=0.1)
ax.view_init(elev=0, azim=-90)
plt.show()

# %%
# Save the new observation points

saved_Fg_file = os.path.join(datapath, 'Fg_triloop_%s_%s_robs_new.npz'%(material, diffraction_plane))
np.savez(saved_Fg_file, r_obs=RL.reshape(-1, 3))