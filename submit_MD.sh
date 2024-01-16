#!/bin/bash
#SBATCH --job-name=dispgrad
#SBATCH -n 300
#SBATCH -t 7-00:00:00
#SBATCH -p mc

# module load viz devel python/3.9.0 py-numpy/1.24.2_py39 py-scipy/1.10.1_py39 py-matplotlib/3.7.1_py39 openmpi py-mpi4py

# mpirun -np 300 python3 test_diamond_DFXM_mpi.py

# wait

# mpirun -np 300 python3 test_diamond_DFXM_mpi.py diamond_MD0_200x100x100
# wait

mpirun -np 300 python3 test_diamond_DFXM_mpi.py diamond_MD20000_189x100x100
wait

# mpirun -np 300 python3 test_diamond_DFXM_mpi.py diamond_MD50000_174x101x100
# wait

# mpirun -np 300 python3 test_diamond_DFXM_mpi.py diamond_MD100000_149x100x101
# wait

# mpirun -np 300 python3 test_diamond_DFXM_mpi.py diamond_MD150000_131x100x104
# wait

# mpirun -np 300 python3 test_diamond_DFXM_mpi.py diamond_MD200000_114x100x107
# wait
