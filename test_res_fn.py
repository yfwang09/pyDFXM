'''
This file is used to test the reciprocal space resolution function

Created on 2023-09-12 9:33 PM

Author: yfwang09

'''

# %%
# Load packages
import os
from re import L
import numpy as np
import forward_model as fwd
import visualize_helper as vis

os.makedirs('data', exist_ok=True)
res_save_file = os.path.join('data', 'Res_qi_Al_001.npz')
saved_q_file = os.path.join('data', 'saved_q_Al_001.npz')

tests = [1, ]

# %%
# Test 1. Test the resolution function in Poulsen et al. (2021)
if 1 in tests:
    print('1. Test the resolution function in Poulsen et al. (2021)')

    forward_dict = fwd.default_forward_dict()
    forward_dict['phi'] = -0.0004 # for benchmarking only, NEVER change phi during actual res_fn call!
    forward_dict['Nrays'] = 10000
    forward_dict['q1_range'] = 2e-3; forward_dict['npoints1'] = 40;
    forward_dict['q2_range'] = 5e-3; forward_dict['npoints2'] = 40;
    forward_dict['q3_range'] = 5e-3; forward_dict['npoints3'] = 40;
    print('The forward model dictionary is:')
    print(forward_dict)
    fwd_model = fwd.DFXM_forward(forward_dict)
    Res_qi, saved_q = fwd_model.res_fn(plot=True, timeit=True)
    figax = vis.visualize_res_fn_slice_z(fwd_model.d, Res_qi)

# %%
# Test 2. Test the resolution function of our own.
if 2 in tests:
    print('2. Test the resolution function of our own.')
    # Test the resolution function
    forward_dict = fwd.default_forward_dict()
    fwd_model = fwd.DFXM_forward(forward_dict)
    Res_qi, saved_q = fwd_model.res_fn(timeit=True)

    vis.visualize_res_fn_slice_z(fwd_model.d, Res_qi)
    if not os.path.exists(res_save_file):
        np.savez_compressed(res_save_file, Res_qi=Res_qi)
    if not os.path.exists(saved_q_file):
        np.savez_compressed(saved_q_file, saved_q=saved_q)

# %%
# Test 3. Test the resolution function of our own with saved q vectors. (fast)
if 3 in tests:
    print('3. Test the resolution function of our own with saved q vectors. (fast)')
    # Test the resolution function
    forward_dict = fwd.default_forward_dict()
    fwd_model = fwd.DFXM_forward(forward_dict)
    saved_q = np.load(saved_q_file)['saved_q']
    Res_qi, _ = fwd_model.res_fn(saved_q=saved_q, timeit=True)

    figax = vis.visualize_res_fn_slice_z(fwd_model.d, Res_qi)

# %%
    
if 4 in tests:
    import matplotlib.pyplot as plt

    mu = np.mean(saved_q, axis=0)
    sigma = np.std(saved_q, axis=0)

    q1vals = np.arange(-forward_dict['q1_range']/2, forward_dict['q1_range']/2, forward_dict['q1_range']/forward_dict['npoints1'])
    q2vals = np.arange(-forward_dict['q2_range']/2, forward_dict['q2_range']/2, forward_dict['q2_range']/forward_dict['npoints2'])
    q3vals = np.arange(-forward_dict['q3_range']/2, forward_dict['q3_range']/2, forward_dict['q3_range']/forward_dict['npoints3'])

    Rq1 = np.exp(-0.5 * (q1vals - mu[0])**2 / sigma[0]**2)
    Rq2 = np.exp(-0.5 * (q2vals - mu[1])**2 / sigma[1]**2)
    Rq3 = np.exp(-0.5 * (q3vals - mu[2])**2 / sigma[2]**2)

    L1, L2, L3 = Res_qi.shape
    fig, ax = plt.subplots()
    ax.plot(q1vals, Res_qi[:, L2//2, L3//2], 'C1', label='qy_center')
    ax.plot(q2vals, Res_qi[L1//2, :, L3//2], 'C2', label='qz_center')
    ax.plot(q3vals, Res_qi[L1//2, L2//2, :], 'C0', label='qx_center')

    ax.plot(q1vals, Rq1, 'C0--', label='Rq1')
    ax.plot(q2vals, Rq2, 'C1--', label='Rq2')
    ax.plot(q3vals, Rq3, 'C2--', label='Rq3')



    ax.legend()
    plt.show()


# %%
