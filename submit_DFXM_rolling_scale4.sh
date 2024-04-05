#!/bin/bash
#SBATCH --job-name=dispgrad
#SBATCH -n 20
#SBATCH --mem 64000
#SBATCH -t 7-00:00:00
#SBATCH -p mc

# Calculate disp grad using mpi4py
# module load viz devel python/3.9.0 py-numpy/1.24.2_py39 py-scipy/1.10.1_py39 py-matplotlib/3.7.1_py39 openmpi py-mpi4py
ml system viz devel git python/3.9.0 py-numpy/1.20.3_py39 py-scipy/1.6.3_py39 py-matplotlib/3.4.2_py39 py-numba/0.54.1_py39 openmpi py-mpi4py

for case in diamond_MD0_200x100x100 diamond_MD20000_189x100x100 diamond_MD50000_174x101x100 diamond_MD100000_149x100x101 diamond_MD150000_131x100x104 diamond_MD200000_114x100x107
# for case in diamond_MD20000_189x100x100 diamond_MD0_200x100x100
# for case in diamond_MD50000_174x101x100 diamond_MD100000_149x100x101 diamond_MD150000_131x100x104 diamond_MD200000_114x100x107
do
    mpirun -np 20 python3 test_diamond_DFXM_mpi_workflow.py --casename ${case} -phi 0 -chi 0 -hkl 111 -sc 0.5
    mpirun -np 20 python3 test_diamond_DFXM_mpi_workflow.py --casename ${case} -phi 0 -chi 0 -hkl 004 -sc 0.25
    wait
    for i in {01..40}
    do
        mpirun -np 20 python3 test_diamond_DFXM_mpi_workflow.py --casename ${case} -chi 0.00${i} -phi 0 -hkl 111 -sc 0.5
        wait
        mpirun -np 20 python3 test_diamond_DFXM_mpi_workflow.py --casename ${case} -chi 0.00${i} -phi 0 -hkl 004 -sc 0.25
        wait
        mpirun -np 20 python3 test_diamond_DFXM_mpi_workflow.py --casename ${case} -chi -0.00${i} -phi 0 -hkl 111 -sc 0.5
        wait
        mpirun -np 20 python3 test_diamond_DFXM_mpi_workflow.py --casename ${case} -chi -0.00${i} -phi 0 -hkl 004 -sc 0.25
        wait
    done
done


# MPI segment calculations

# mpirun -np 300 python3 test_diamond_DFXM_mpi.py
# wait

# mpirun -np 300 python3 test_diamond_DFXM_mpi.py diamond_MD0_200x100x100
# wait

# rm -rfv data/Fg_diamond_MD20000_189x100x100_seg
# mpirun -np 300 python3 test_diamond_DFXM_mpi.py diamond_MD20000_189x100x100
# wait

# mpirun -np 300 python3 test_diamond_DFXM_mpi.py diamond_MD50000_174x101x100
# wait

# mpirun -np 300 python3 test_diamond_DFXM_mpi.py diamond_MD100000_149x100x101
# wait

# mpirun -np 300 python3 test_diamond_DFXM_mpi.py diamond_MD150000_131x100x104
# wait

# mpirun -np 300 python3 test_diamond_DFXM_mpi.py diamond_MD200000_114x100x107
# wait
