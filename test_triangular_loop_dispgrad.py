#%%----------------------------------------------------------------------
# Non-singular displacement gradient for a triangular dislocation loop
#
# This notebook tests the non-singular displacement gradient for
# a triangular dislocation loop. The non-singular displacement gradient is
# given by the DDD-XRD code, and is validated against the analytical 
# expression provided in Bertin and Cai, CMS, (2018).
#
# In this code all the length are normalized to the unit of Burger's vector.
# Note that all the rn and r are in the unit of Burger's vector.
# The bmag is only used in the Fg function for DFXM calculation 
# (in test_triangular_loop_DFXM.py)
#
# This code does the following:
# * generates a random triangular dislocation loop
# * computes the displacement gradient field along a line using the non-singular expression provided in Bertin and Cai, CMS, 2018.
# * compares it with the displacement gradient field obtained by numerically differentiating the displacement field
# * compares the stress obtained from the displacement gradient field with the non-singular stress expression provided in Cai et al., 2006
# 
# This code uses the DDLab data structure to represent the dislocations.
# 
# Original DDD-XRD: Nicolas Bertin, nbertin@stanford.edu
# Python translation: Yifan Wang, yfwang09@stanford.edu
#---------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import dispgrad_func as dgf
import time
from displacement_grad_helper import displacement_structure, displacement_gradient_structure_matlab

#---------------------------------------------------------
# INITIALIZATION
#---------------------------------------------------------

# Elasticity parameters (Aluminum)
input_dict = dgf.default_dispgrad_dict('disl_network')

MU = 27e9                           # Shear modulus (Pa)
input_dict['nu'] = NU = 0.324       # Poisson's ratio
LAMBDA = 2*MU*NU/(1-2*NU)           # Lame's first parameter
input_dict['b'] = bmag = 1.0        # Burger's vector magnitude
input_dict['a'] = a = 1.0           # Non-singular radius (in the unit of Burger's vector)

# Initialize the triangular loop 
L = 1000        # in the unit of b
rn = L*(np.random.rand(3, 3) - 0.5)
rn = np.array([[ 78.12212123, 884.74707189, 483.30385117],
               [902.71333272, 568.95913492, 938.59105117],
               [500.52731411, 261.22281654, 552.66098404]]) - L/2
print(rn)
# Normalized Burger's vector
b = np.array([1, 1, 0])
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

#---------------------------------------------------------
# Plot the dislocation loop
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(links.shape[0]):
    n12 = links[i, 0:2].astype(int)
    r12 = rn[n12, :]
    ax.plot(r12[..., 0], r12[..., 1], r12[..., 2],  'C3o-')

#---------------------------------------------------------
# EVALUATE DISPLACEMENT GRADIENT FIELD
#---------------------------------------------------------

# Evaluate strain gradient from non-singular expression 
# along a x-line passing through the center of the loop
N = 50 # Number of points along the line
N = 10000
r = np.zeros((N, 3))
r[:, 0] = np.linspace(0, L, N)
r[:, 1] = np.mean(rn[:, 1])
r[:, 2] = np.mean(rn[:, 2])
ax.plot(r[:, 0], r[:, 1], r[:, 2], 'C0o-', label='field points')
plt.legend()

# Evaluate the displacement gradient field using the analytical expression (Bertin and Cai, CMS, 2018)
# The length is normalized to the unit of Burger's vector (bmag)
tic = time.time()
dudx = disl.displacement_gradient_structure(r)
toc = time.time()
print('Time to evaluate displacement gradient field: ', toc-tic)
tic = time.time()
dudx_ref = displacement_gradient_structure_matlab(rn, links, NU, a, r)
toc = time.time()
print('Time to evaluate displacement gradient field (reference): ', toc-tic)
tic = time.time()
dudx_ref = displacement_gradient_structure_matlab(rn, links, NU, a, r)
toc = time.time()
print('Time to evaluate displacement gradient field (jit): ', toc-tic)
print('dudx error: ', np.linalg.norm(dudx - dudx_ref))

#%%-------------------------------------------------------
# Evaluate displacement gradient by numerically differentiating the displacement field
delta = 2
rd = r.copy()
rd[:, 0] = rd[:, 0] - delta
u10 = displacement_structure(rn, links, NU, a, rd)
rd = r.copy()
rd[:, 0] = rd[:, 0] + delta
u11 = displacement_structure(rn, links, NU, a, rd)
dudx1 = (u11 - u10)/2/delta
rd = r.copy()
rd[:, 1] = rd[:, 1] - delta
u20 = displacement_structure(rn, links, NU, a, rd)
rd = r.copy()
rd[:, 1] = rd[:, 1] + delta
u21 = displacement_structure(rn, links, NU, a, rd)
dudx2 = (u21 - u20)/2/delta
rd = r.copy()
rd[:, 2] = rd[:, 2] - delta
u30 = displacement_structure(rn, links, NU, a, rd)
rd = r.copy()
rd[:, 2] = rd[:, 2] + delta
u31 = displacement_structure(rn, links, NU, a, rd)
dudx3 = (u31 - u30)/2/delta
dudx_list = [dudx1, dudx2, dudx3]

# Plot and compare displacement gradients
fig, axs = plt.subplots(3, 3)
for i in range(3):
    for j in range(3):
        ax = axs[i, j]
        ax.plot(r[:, 0], dudx[:, i, j], '-k')
        ax.plot(r[:, 0], dudx_list[j][:, i], '+r')
        ax.set_title(r'$du_{}/dx_{}$'.format(i+1, j+1))

plt.show()
