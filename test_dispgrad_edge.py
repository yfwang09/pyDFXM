#-----------------------------------------------------------------------
# DFXM forward calculation for a straight edge dislocation
#
# Developer: Yifan Wang, yfwang09@stanford.edu
# Date: 2023/12/27
#---------------------------------------------------------------

perform_test = [1, 2, 3, 4, 5]

# This code tests the the displacement gradient field of a single edge dislocation

print('------- Test 1: Compute the rotation matrix from function --------')

if 1 not in perform_test:
    print('Test 1 skipped')
else:

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

if 2 not in perform_test:
    print('Test 2 skipped')
else:
    import numpy as np
    import matplotlib.pyplot as plt
    import dispgrad_func as dgf

    print('Create the strain field of an edge dislocation')

    xd = np.linspace(-1, 1, 100)
    yd = np.linspace(-1, 1, 100)
    input_dict = {
        'b': 1, 'nu': 0.334,
        'bg': [1,-1, 0], # Burger's vector dir. in Miller (grain coord)
        'ng': [1, 1,-1], # Normal vector dir. in Miller (grain coord)
        'tg': [1, 1, 2], # Dislocation line dir. in Miller (grain coord)
    }
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
            ax.set_title(r'$H^d_{%s%s}$'%(subs[i], subs[j]), fontsize=fs)

    fig.colorbar(cim, ax=axs)
    fig.suptitle(r'Dislocation coordinate system $\mathbf{H}^d=\mathbf{F}^d-\mathbf{I}$')
    plt.show()

print('------------------------------------------------------------------')
print('-------- Test 3: 3D strain field of an edge dislocation ----------')

if 3 not in perform_test:
    print('Test 3 skipped')
else:
    import numpy as np
    import matplotlib.pyplot as plt
    import dispgrad_func as dgf
    import forward_model as fwd
    import visualize_helper as vis

    print('Set up of the dislocation structure')
    # The dispgrad_func class always assuming the coordinate system is aligned with Miller's indices (grain coordinate).
    input_dict = {
        'b': 1, 'nu': 0.334,
        # The following defines the dislocation characters
        'bg': [1,-1, 0], # Burger's vector dir. in Miller (grain)
        'ng': [1, 1,-1], # Normal vector dir. in Miller (grain)
        'tg': [1, 1, 2], # Dislocation line dir. in Miller (grain)
    }
    print(input_dict)
    edge = dgf.edge_disl(input_dict)
    Ud = edge.Ud
    print('Ud =')
    print(Ud)

    forward_dict = fwd.default_forward_dict()
    # The following defines the sample coordinate (Ug, Ug = U.T, Eq.7-8)
    forward_dict['x_c'] = [1,0,0]  # x dir. for the crystal system (Fig.2)
    forward_dict['y_c'] = [0,1,0]  # y dir. for the crystal system (Fig.2)
    forward_dict['hkl'] = [0,0,1]  # hkl diffraction plane, z dir. crystal
    # forward_dict['hkl'] = [1, 1, 1]
    # forward_dict['x_c'] = [1, 1,-2]
    # forward_dict['y_c'] = [-1,1, 0]
    # forward_dict['hkl'] = [1, 1,-1]
    # forward_dict['x_c'] = [1, 1, 2]
    # forward_dict['y_c'] = [1,-1, 0]
    model = fwd.DFXM_forward(forward_dict, load_res_fn=None)
    Ug = model.Ug

    print('Define the voxelized 3D field')

    lb, ub = -1, 1
    Ngrid = 50
    offset = 1e-4                           # to avoid the singularity at the core
    xs = np.linspace(lb, ub, Ngrid)-offset  # (Ngrid, )
    ys = np.linspace(lb, ub, Ngrid)         # (Ngrid, )
    zs = np.linspace(lb, ub, Ngrid)         # (Ngrid, )

    XX, YY, ZZ = np.meshgrid(xs, ys, zs)    # (Ngrid, Ngrid, Ngrid)
    Rs = np.stack([XX,YY,ZZ], -1)           # (Ngrid, Ngrid, Ngird, 3)
    print('Grid size in the sample coordinates:', Rs.shape)

    print('Convert Rs into the grain coordinates (Miller indices)')
    Rg = np.einsum('ij,...j->...i', Ug, Rs)

    print('Rotate the grid into the dislocation coordinate')
    Rd = np.einsum('ij,...j->...i', Ud.T, Rg)    # (Ngrid, Ngrid, Ngrid, 3)
    # Rd = np.sum(np.linalg.inv(Ud)*RR[:,:,:,None,:], axis=-1)
    Rdx = Rd[..., 0]
    Rdy = Rd[..., 1]

    print('2D dislocation coordinate:', Rdx.shape, Rdy.shape)

    print('Calculate the strain tensor')
    Fd = edge.get_disl_strain_tensor(Rdx, Rdy)      # (Ngrid, Ngrid, Ngrid, 3, 3)
    print('Strain tensor:', Fd.shape)

    print('Convert back to the grain coordinate (Miller)')
    Fg = np.einsum('ij,...jk,kl->...il', Ud, Fd, Ud.T)         # (Ngrid, Ngrid, Ngrid, 3, 3)
    print('Fg.shape =', Fg.shape)

    print('Convert back to the sample coordinate')
    Fs = np.einsum('ij,...jk,kl->...il', Ug.T, Fg, Ug)         # (Ngrid, Ngrid, Ngrid, 3, 3)
    print('Fs.shape =', Fs.shape)

    subs = ['x', 'y', 'z']
    vmin, vmax = -1, 1
    F_plot = Fs - np.identity(3)

    fs = 12

    if 3.1 in perform_test:
        print('Visualize the Fs tensor as 2D slices')

        for iz in np.linspace(0, Ngrid - 1, 9).round().astype(int):
            fig, axs = plt.subplots(3, 3, figsize=(6, 4),
                                    sharex=True, sharey=True)
            for i in range(3):
                for j in range(3):
                    F_z = F_plot[:, :, iz, i, j]
                    ax = axs[i][j]
                    im = ax.imshow(F_z, extent=[lb, ub, lb, ub], vmin=vmin, vmax=vmax, origin='lower')
                    ax.set_title(r' $H^g_{%s%s}$'%(subs[i], subs[j]), loc='left', y=0.6, color='w')
                axs[i][0].set_ylabel(r'$y^s/b$')
                axs[-1][i].set_xlabel(r'$x^s/b$')
            fig.colorbar(im, ax=axs)
            fig.suptitle(r'Grain coordinate system, $z^s/b$=%.4f'%zs[iz], fontsize=fs)
            print('  visualizing the slice zs = %.4f'%zs[iz])
        plt.show()

    print('Test the visualization helper function plot_2d_slice_z()')
    extent = [lb, ub, lb, ub, lb, ub]
    figax = vis.plot_2d_slice_z(F_plot[:, :, :, 1, 2], extent=extent, vmin=vmin, vmax=vmax, show=False)
    figax[0].suptitle(r'Grain coordinate system, $H^s_{yz}$', fontsize=fs)
    # plt.show()

    print('Visualize 3D slices of each component in Hg')

    from matplotlib.colors import Normalize
    subs = ['x', 'y', 'z']

    print('Define the x,y coordinates of the slices')
    xx, yy = np.meshgrid(xs, ys)
    zz = (-edge.ng[0] * xx - edge.ng[1] * yy) / edge.ng[2]
    zz[zz>ub] = np.nan
    zz[zz<lb] = np.nan
    vmin, vmax = -1, 1
    fs = 12

    print('Define the x,y,z coordinates of the dislocation line')
    ztg = np.linspace(lb, ub)
    ytg = ztg/edge.tg[2]*edge.tg[1]
    xtg = ztg/edge.tg[2]*edge.tg[0]
    # Convert from grain to sample coordinates
    rts = np.einsum('ij,...j->...i', (Ug.T), np.transpose([xtg, ytg, ztg]))
    xt, yt, zt = rts[..., 0], rts[..., 1], rts[..., 2]

    if 3.3 in perform_test:
        print('Visualize the 3D slices')
        for i in range(3):
            for j in range(3):
                print('  visualizing %s%s component'%(subs[i], subs[j]))
                fig = plt.figure(figsize=(6, 4))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot([lb, lb, ub, ub], [ub, ub, ub, ub], [lb, ub, ub, lb], 'k')
                ax.plot([lb, lb, ub, ub], [lb, ub, ub, lb], [lb, lb, lb, lb], 'k')
                ax.plot(xt, yt, zt, 'k', linewidth=2)

                for iz in range(5, 50, 10):
                    zloc = zs[iz]                  # z location of the plane
                    Fg_z = F_plot[:, :, iz, i, j]      # value of the plane
                    zval = np.clip(Fg_z, vmin, vmax)
                    # print(zval.min(), zval.max())
                    norm = Normalize(vmin=vmin, vmax=vmax)
                    surf = ax.plot_surface(xx, yy, np.ones_like(xx)*zloc, linewidth=0, edgecolor="None", facecolors=plt.cm.viridis(norm(zval)), alpha=0.5)
                    
                cax = fig.colorbar(surf, shrink=0.5)
                cticks = cax.get_ticks()  # Always in the range of [0, 1]
                cax.set_ticks(cticks)
                cticklabels = (cticks-cticks.min())/(cticks.max()-cticks.min())*(vmax-vmin) + vmin
                cax.set_ticklabels(['%.1f'%k for k in cticklabels])
                ax.set_title(r'Sample coordinate system $H^s_{%s%s}$'%(subs[i], subs[j]), fontsize=fs)

                ax.plot([lb, lb, ub, ub], [lb, lb, lb, lb], [ub, lb, lb, ub], 'k')
                ax.plot([lb, lb, ub, ub], [ub, lb, lb, ub], [ub, ub, ub, ub], 'k')
                ax.set_xlabel(r'$x^s/b$', fontsize=fs)
                ax.set_ylabel(r'$y^s/b$', fontsize=fs)
                ax.set_zlabel(r'$z^s/b$', fontsize=fs)
        plt.show()

    print('Test the visualization helper function plot_3d_slice_z')
    figax = vis.plot_3d_slice_z(F_plot[:, :, :, 1, 2], extent=[lb, ub, lb, ub, lb, ub], vmin=vmin, vmax=vmax, nslice=5, fs=fs, show=False)
    fig, ax = figax
    # change the view angle
    # ax.view_init(elev=90, azim=-90)
    ax.plot(xt, yt, zt, 'k', lw=2)
    ax.set_title(r'Grain coordinate system, $H^s_{yz}$', fontsize=fs)
    plt.show()

print('------------------------------------------------------------------')
print('-------- Test 4: DFXM image of an edge dislocation ----------')

if 4 not in perform_test:
    print('Test 4 skipped')
else:
    import numpy as np
    import matplotlib.pyplot as plt
    import dispgrad_func as dgf
    import forward_model as fwd
    import visualize_helper as vis

    print('Set up of the dislocation structure')
    b0 = 2.86e-10 # Burgers vector of Aluminum in m 
    # The dispgrad_func class always assuming the structure is aligned with Miller's indices.
    input_dict = {
        'b': b0, 'nu': 0.334,
        # The following defines the dislocation characters
        'bg': [1,-1, 0], # Burger's vector dir. in Miller (grain)
        'ng': [1, 1,-1], # Normal vector dir. in Miller (grain)
        'tg': [1, 1, 2], # Dislocation line dir. in Miller (grain)
    }
    print(input_dict)
    edge = dgf.edge_disl(input_dict)

    forward_dict = fwd.default_forward_dict()
    # The following defines the sample coordinate (Ug, Ug = U.T, Eq.7-8)
    forward_dict['x_c'] = [1, 0, 0]  # x dir. for the crystal system (Fig.2)
    forward_dict['y_c'] = [0, 1, 0]  # y dir. for the crystal system (Fig.2)
    forward_dict['hkl'] = [0, 0, 1]  # hkl diffraction plane, z dir. crystal
    forward_dict['Npixels'] = [50, 50, 50]
    model = fwd.DFXM_forward(forward_dict, load_res_fn='data/Res_qi_Al_001.npz')

    # Calculate and visualize the image
    print('#'*20 + ' Calculate the strong-beam DFXM image ' + '#'*20)
    im, ql, rulers = model.forward(edge.Fg)
    print('im.shape =', im.shape)
    print('ql.shape =', ql.shape)
    print('chi =', np.rad2deg(forward_dict['chi']), 'phi =', np.rad2deg(forward_dict['phi']))  # tilt angle in deg

    # Visualize the simulated image
    figax = vis.visualize_im_qi(forward_dict, im, None, rulers, unit='um', deg=True, show=False)

    # Visualize the reciprocal space wave vector ql
    # figax = vis.visualize_im_qi(forward_dict, None, ql, rulers, vlim_qi=[-1e-10, 1e-10], unit='um', deg=True)
    figax = vis.visualize_im_qi(forward_dict, None, ql, rulers, vlim_qi=[-1e-4, 1e-4])

    # Calculate and visualize the image
    print('#'*20 + ' Calculate the weak-beam DFXM image ' + '#'*20)
    forward_dict['phi'] = np.deg2rad(0.05)  # tilt angle in rad
    forward_dict['chi'] = np.deg2rad(0.000)  # tilt angle in rad
    im, ql, rulers = model.forward(edge.Fg)
    print('im.shape =', im.shape)
    print('ql.shape =', ql.shape)
    print('chi =', np.rad2deg(forward_dict['chi']), 'phi =', np.rad2deg(forward_dict['phi']))  # tilt angle in deg

    # Visualize the simulated image
    figax = vis.visualize_im_qi(forward_dict, im, None, rulers, unit='um', deg=True, show=False)

    # Visualize the reciprocal space wave vector ql
    # figax = vis.visualize_im_qi(forward_dict, None, ql, rulers, vlim_qi=[-1e-10, 1e-10], unit='um', deg=True)
    figax = vis.visualize_im_qi(forward_dict, None, ql, rulers, vlim_qi=[-1e-4, 1e-4])

print('------------------------------------------------------------------')
print('-------- Test 5: Rocking curve of the DFXM for an edge -----------')

if 5 not in perform_test:
    print('Test 5 skipped')
else:
    import numpy as np
    import matplotlib.pyplot as plt
    import dispgrad_func as dgf
    import forward_model as fwd
    import visualize_helper as vis

    print('Set up of the dislocation structure')
    b0 = 2.86e-10 # Burgers vector of Aluminum in m
    # The dispgrad_func class always assuming the structure is aligned with Miller's indices.
    input_dict = {
        'b': b0, 'nu': 0.334,
        # The following defines the dislocation characters
        'bg': [1,-1, 0], # Burger's vector dir. in Miller (grain)
        'ng': [1, 1,-1], # Normal vector dir. in Miller (grain)
        'tg': [1, 1, 2], # Dislocation line dir. in Miller (grain)
    }
    print(input_dict)
    edge = dgf.edge_disl(input_dict)

    forward_dict = fwd.default_forward_dict()
    # The following defines the sample coordinate (Ug, Ug = U.T, Eq.7-8)
    forward_dict['x_c'] = [1, 0, 0]  # x dir. for the crystal system (Fig.2)
    forward_dict['y_c'] = [0, 1, 0]  # y dir. for the crystal system (Fig.2)
    forward_dict['hkl'] = [0, 0, 1]  # hkl diffraction plane, z dir. crystal
    forward_dict['Npixels'] = [50, 50, 50]
    model = fwd.DFXM_forward(forward_dict, load_res_fn='data/Res_qi_Al_001.npz')

    # Calculate the rocking curve
    print('#'*20 + ' Calculate the rocking curve ' + '#'*20)
    phi_values = np.round(np.arange(-0.0020, 0.0022, 0.0002), 4)
    rocking_maxvals = []
    for i, phi in enumerate(phi_values):
        forward_dict['phi'] = phi
        im, ql, rulers = model.forward(edge.Fg)
        print('chi =', np.rad2deg(forward_dict['chi']), 'phi =', np.rad2deg(forward_dict['phi']))  # tilt angle in deg

        if i%5 == 0:
            # Visualize the simulated image
            figax = vis.visualize_im_qi(forward_dict, im, None, rulers, unit='um', deg=True, show=False)

        rocking_maxvals.append(im.max())

    # Visualize the rocking curve
    fig, ax = plt.subplots()
    ax.plot(np.rad2deg(phi_values), rocking_maxvals, 'k')
    ax.set_xlabel(r'$\phi (\degree)$')
    ax.set_ylabel('Maximum intensity')
    plt.show()

print('------------------------------------------------------------------')