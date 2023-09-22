'''
This file is used to test the reciprocal space resolution function

Created on 2023-09-12 9:33 PM

Author: yfwang09

'''

import os
import numpy as np
import forward_model as fwd
import visualize_helper as vis

os.makedirs('data', exist_ok=True)
res_save_file = os.path.join('data', 'Res_qi_Al001.npz')

print('1. Test the resolution function in Poulsen et al. (2021)')

forward_dict = fwd.default_forward_dict()
forward_dict['phi'] = -0.0004 # for benchmarking only, NEVER change phi during actual res_fn call!
forward_dict['Nrays'] = 10000
forward_dict['q1_range'] = 2e-3; forward_dict['npoints1'] = 40;
forward_dict['q1_range'] = 5e-3; forward_dict['npoints1'] = 40;
forward_dict['q1_range'] = 5e-3; forward_dict['npoints1'] = 40;
print('The forward model dictionary is:')
print(forward_dict)
fwd_model = fwd.DFXM_forward(forward_dict)
Res_qi, saved_q = fwd_model.res_fn(plot=True, timeit=True)
vis.visualize_res_fn_slice_z(fwd_model.d, Res_qi)

print('2. Test the resolution function of our own.')
# Test the resolution function
forward_dict = fwd.default_forward_dict()
fwd_model = fwd.DFXM_forward(forward_dict)
Res_qi, saved_q = fwd_model.res_fn(timeit=True)

vis.visualize_res_fn_slice_z(fwd_model.d, Res_qi)
np.savez_compressed(res_save_file, Res_qi=Res_qi, saved_q=saved_q)

print('3. Test the resolution function of our own with saved q vectors. (fast)')
# Test the resolution function
forward_dict = fwd.default_forward_dict()
fwd_model = fwd.DFXM_forward(forward_dict)
saved_q = np.load(res_save_file)['saved_q']
Res_qi, _ = fwd_model.res_fn(saved_q=saved_q, timeit=True)

vis.visualize_res_fn_slice_z(fwd_model.d, Res_qi)