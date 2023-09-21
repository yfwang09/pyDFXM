'''
This file is used to test the reciprocal space resolution function

Created on 2023-09-12 9:33 PM

Author: yfwang09

'''

import numpy as np
from forward_model import DFXM_forward

# Test the resolution function
fwd = DFXM_forward()
Res_qi, saved_q = fwd.res_fn(timeit=True)

