#-----------------------------------------------------------------------
# DFXM forward calculation for the simple shear case
#
# Developer: Yifan Wang, yfwang09@stanford.edu
# Date: 2023/12/27
#---------------------------------------------------------------

import os
import numpy as np
import dispgrad_func as dgf
import forward_model as fwd
import visualize_helper as vis

# Set up the displacement gradient field function
print('#'*20 + ' Set up the displacement gradient field function')
input_dict = dgf.default_dispgrad_dict('simple_shear')
print(input_dict)
shear = dgf.simple_shear(input_dict)

# Set up the forward model
print('#'*20 + ' Set up the forward model')
forward_dict = fwd.default_forward_dict()
print(forward_dict)

datapath = 'data'
os.makedirs(datapath, exist_ok=True)
saved_res_fn = os.path.join(datapath, 'Res_qi_Al_001.npz')
print('saved resolution function at %s'%saved_res_fn)

model = fwd.DFXM_forward(forward_dict, load_res_fn=saved_res_fn)

# Calculate and visualize the image
print('#'*20 + ' Calculate and visualize the image')
im, ql, rulers = model.forward(shear.Fg)

# Visualize the simulated image
figax = vis.visualize_im_qi(forward_dict, im, None, rulers, vlim_im=[0, 200])

# Visualize the reciprocal space wave vector ql
figax = vis.visualize_im_qi(forward_dict, None, ql, rulers, vlim_qi=[-1e-4, 1e-4])
