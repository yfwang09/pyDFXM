# This code tests the the displacement gradient field of a single edge dislocation

print('------- Test 1: Compute the rotation matrix from function --------')

from matplotlib import tight_layout
import numpy as np
import dispgrad_func as dgf

be = [1,-1, 0]
ns = [1, 1,-1]
xi = [1, 1, 2]

print('Burgers vector', be)
print('Normal vector', ns)
print('Line direction', xi)
print()

# Calculate the rotation matrix based on the normal and dislocation line directions (equation 65 in Poulsen et al., 2021)
print('Ud from return_dis_grain_matrices')
Ud = dgf.return_dis_grain_matrices(b=be, n=ns, t=xi)
print(Ud)

# Compare with the original all matrices in the all 12 slip systems (transposed)
print('Ud from slip system database')
Udall = dgf.return_dis_grain_matrices_all()
print(Udall[:, :, 3].T)

test_pass = np.linalg.norm((Udall[:, :, 3].T - Ud).flatten()) < 1e-7
print('Test passed?', test_pass)

print('------------------------------------------------------------------')

print('-------- Test 2: 2D strain field of an edge dislocation ----------')

import numpy as np
import matplotlib.pyplot as plt
import dispgrad_func as dgf

print('Create the strain field of an edge dislocation')

xd = np.linspace(-1, 1, 100)
yd = np.linspace(-1, 1, 100)
input_dict = {'b': 1, 'nu': 0.334}
print(input_dict)

edge = dgf.edge_disl(input_dict)

XX, YY = np.meshgrid(xd, yd)
Hd = edge.get_disl_strain_tensor(XX, YY) - np.identity(3)
print('Fd - I = Hd.shape =', Hd.shape)

subs = ['x', 'y', 'z']
strain = (Hd + np.swapaxes(Hd, -1, -2))/2
print('Visualize the strain tensor in plane')

fs = 12
vmin, vmax = -0.4, 0.4
fig, axs = plt.subplots(2, 2, figsize=(6, 5), 
                        sharex=True, sharey=True, )
for i in range(2):
    for j in range(2):
        ax = axs[i][j]
        cim = ax.imshow(Hd[..., i,j], extent=[-1,1,-1,1], vmin=vmin, vmax=vmax)
        if i == 1:
            ax.set_xlabel('x/b', fontsize=fs)
        if j == 0:
            ax.set_ylabel('y/b', fontsize=fs)
        ax.set_title(r'$H_{%s%s}$'%(subs[i], subs[j]), fontsize=fs)

fig.colorbar(cim, ax=axs)
fig.suptitle(r'$\mathbf{H}=\mathbf{F-I}$')
plt.show()

print('------------------------------------------------------------------')

print('-------- Test 3: 3D strain field of an edge dislocation ----------')

import numpy as np
import matplotlib.pyplot as plt
import dispgrad_func as dgf
import visualize_helper as vis

print('Set up of the dislocation structure')
input_dict = {'b': 1, 'nu': 0.334, 
              # The following defines the grain rotation (Ug, Eq.7-8)
              'hkl': [0,0,1], # hkl diffraction plane, z dir. crystal
              'x_c': [1,0,0], # x dir. for the crystal system (Fig.2)
              'y_c': [0,1,0], # y dir. for the crystal system (Fig.2)
              # The following defines the dislocation structure
              'bs': [1,-1, 0], # Burger's vector dir. in Miller (sample)
              'ns': [1, 1,-1], # Normal vector dir. in Miller (sample)
              'ts': [1, 1, 2], # Dislocation line dir. in Miller (sample)
             }
print(input_dict)

print('Define the voxelized 3D field')

lb, ub = -1, 1
Ngrid = 50
offset = 1e-4                           # to avoid the singularity at the core
xg = np.linspace(lb, ub, Ngrid)-offset  # (Ngrid, )
yg = np.linspace(lb, ub, Ngrid)         # (Ngrid, )
zg = np.linspace(lb, ub, Ngrid)         # (Ngrid, )

XX, YY, ZZ = np.meshgrid(xg, yg, zg)    # (Ngrid, Ngrid, Ngrid)
RR = np.stack([XX,YY,ZZ], -1)           # (Ngrid, Ngrid, Ngird, 3)
print('Grid size:', RR.shape)

print('Rotate the grid into the dislocation coordinate')
Rd = np.sum(np.linalg.inv(Ud)*RR[:,:,:,None,:], axis=-1)    # (Ngrid, Ngrid, Ngrid, 3)
Rdx = Rd[..., 0]
Rdy = Rd[..., 1]

print('2D dislocation coordinate:', Rdx.shape, Rdy.shape)

print('Calculate the strain tensor')
edge = dgf.edge_disl(input_dict)
Fd = edge.get_disl_strain_tensor(Rdx, Rdy)      # (Ngrid, Ngrid, Ngrid, 3, 3)
print('Strain tensor:', Fd.shape)

print('Convert back to the sample coordinate (Miller)')
Ud = edge.Ud
print('Ud =')
print(Ud)
Fs = np.einsum('ij,...jk,kl->...il', Ud, Fd, Ud.T)         # (Ngrid, Ngrid, Ngrid, 3, 3)
print('Fs.shape =', Fs.shape)

print('Convert back to the grain coordinate')
Ug = edge.Ug
print('Ug =')
print(Ug)
Fg = np.einsum('ij,...jk,kl->...il', Ug, Fs, Ug.T)         # (Ngrid, Ngrid, Ngrid, 3, 3)
print('Fg.shape =', Fg.shape)

print('Visualize the Hg tensor as 2D slices')

subs = ['x', 'y', 'z']
vmin, vmax = -1, 1
Hg = Fg - np.identity(3)

for iz in np.linspace(0, Ngrid - 1, 9).round().astype(int):
    fig, axs = plt.subplots(3, 3, figsize=(6, 4),
                            sharex=True, sharey=True)
    for i in range(3):
        for j in range(3):
            Fg_z = Hg[::-1, :, iz, i, j]
            ax = axs[i][j]
            im = ax.imshow(Fg_z, extent=[lb, ub, lb, ub], vmin=vmin, vmax=vmax)
            ax.set_title(r' $H^g_{%s%s}$'%(subs[i], subs[j]), loc='left', y=0.6, color='w')
        axs[i][0].set_ylabel('y/b')
        axs[-1][i].set_xlabel('x/b')
    fig.colorbar(im, ax=axs)
    fig.suptitle('z=%.4f'%zg[iz], fontsize=fs)
    print('  visualizing the slice z = %.4f'%zg[iz])
plt.show()

print('Visualize 3D slices of each component in Hg')

from matplotlib.colors import Normalize
subs = ['x', 'y', 'z']

print('Define the x,y coordinates of the slices')
xx, yy = np.meshgrid(xg, yg)
zz = (-ns[0] * xx - ns[1] * yy) / ns[2]
zz[zz>ub] = np.nan
zz[zz<lb] = np.nan
vmin, vmax = -0.4, 0.4
fs = 12

print('Define the x,y,z coordinates of the dislocation line')
zt = np.linspace(lb, ub)
yt = zt/xi[2]*xi[1]
xt = zt/xi[2]*xi[0]

for i in range(3):
    for j in range(3):
        print('  visualizing %s%s component'%(subs[i], subs[j]))
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot([lb, lb, ub, ub], [ub, ub, ub, ub], [lb, ub, ub, lb], 'k')
        ax.plot([lb, lb, ub, ub], [lb, ub, ub, lb], [lb, lb, lb, lb], 'k')
        ax.plot(xt, yt, zt, 'k', linewidth=2)
        
        for iz in range(5, 50, 10):
            zloc = zg[iz]                  # z location of the plane
            Fg_z = Hg[:, :, iz, i, j]      # value of the plane
            zval = np.clip(Fg_z, vmin, vmax)
            # print(zval.min(), zval.max())
            norm = Normalize(vmin=vmin, vmax=vmax)
            surf = ax.plot_surface(xx, yy, np.ones_like(xx)*zloc, linewidth=0, edgecolor="None", facecolors=plt.cm.viridis(norm(zval)), alpha=0.3)
            
        cax = fig.colorbar(surf, shrink=0.5)
        cticks = cax.get_ticks()
        # print(cticks)
        cax.set_ticks(cticks)
        # cticklabels = cticks*
        cax.set_ticklabels(['%.1f'%((k*2-1)*vmax) for k in cticks])
        ax.set_title('$F_{%s%s}$'%(subs[i], subs[j]), fontsize=fs)

        ax.plot([lb, lb, ub, ub], [lb, lb, lb, lb], [ub, lb, lb, ub], 'k')
        ax.plot([lb, lb, ub, ub], [ub, lb, lb, ub], [ub, ub, ub, ub], 'k')
        ax.set_xlabel('x/b', fontsize=fs)
        ax.set_ylabel('y/b', fontsize=fs)
        ax.set_zlabel('z/b', fontsize=fs)
plt.show()

print('------------------------------------------------------------------')
