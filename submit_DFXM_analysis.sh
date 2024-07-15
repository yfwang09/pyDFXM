#!/bin/bash
#SBATCH --job-name=dispgrad
#SBATCH -n 56
#SBATCH --mem 64000
#SBATCH -t 7-00:00:00
#SBATCH -p mc

# Calculate disp grad using mpi4py
ml system viz devel git python/3.9.0 py-numpy/1.20.3_py39 py-scipy/1.6.3_py39 py-matplotlib/3.4.2_py39 py-numba/0.54.1_py39 py-pandas/1.3.1_py39 gcc/10.1.0 openmpi/4.1.2 py-mpi4py/3.1.3_py39

for case in diamond_MD20000_189x100x100 diamond_MD50000_174x101x100 diamond_MD100000_149x100x101 diamond_MD150000_131x100x104 diamond_MD200000_114x100x107 diamond_MD0_200x100x100 
do
    for hkl in 400 040 004 -111 1-11 11-1 111
    do
        python3 test_diamond_DFXM_workflow.py --casename ${case} --scale_cell 0.5 -hkl ${hkl} &

        sleep 30

        for slip in 0 1 2 3 4 5 6
        do
            echo "===================================================="
            echo "run slip system ${slip}"
            echo "===================================================="

            python3 test_diamond_DFXM_workflow.py --casename ${case} --scale_cell 0.5 --slip ${slip} -hkl ${hkl} &
        done

        # python3 test_diamond_DFXM_workflow.py --casename ${case} --scale_cell 1.0 --slip ${slip} -hkl 111 &
        # python3 test_diamond_DFXM_workflow.py --casename ${case} --scale_cell 0.5 --slip ${slip} -hkl 111 &
        # python3 test_diamond_DFXM_workflow.py --casename ${case} --scale_cell 1.0 --slip ${slip} -hkl 004 &
        # python3 test_diamond_DFXM_workflow.py --casename ${case} --scale_cell 0.5 --slip ${slip} -hkl 004 &
    done

    # python3 test_diamond_DFXM_workflow.py --casename ${case} --scale_cell 1.0 -hkl 111 &
    # python3 test_diamond_DFXM_workflow.py --casename ${case} --scale_cell 0.5 -hkl 111 &
    # python3 test_diamond_DFXM_workflow.py --casename ${case} --scale_cell 1.0 -hkl 004 &
    # python3 test_diamond_DFXM_workflow.py --casename ${case} --scale_cell 0.5 -hkl 004
    
    wait
done

wait
