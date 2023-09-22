'''
This file is used to test the reciprocal space resolution function

Created on 2023-09-12 9:33 PM

Author: yfwang09

'''

import numpy as np
from forward_model import DFXM_forward
from visualize_helper import visualize_res_fn_slice_z

# Test the resolution function
fwd = DFXM_forward()
Res_qi, saved_q = fwd.res_fn(timeit=True)

visualize_res_fn_slice_z(fwd.d, Res_qi)
