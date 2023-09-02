# This code tests the the displacement gradient field of a single edge dislocation

import numpy as np
import dispgrad_func as dgf
# from dispgrad_func import edge_disl as edge

print('------- Test 1: Compute the rotation matrix from function --------')

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

print('------------------------------------------------------------------')

