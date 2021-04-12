# Using Puhti

This document contains brief instructions relevant to the course. For
general instructions see [CSC User documentation](https://docs.csc.fi)
and the [Puhti quick start guide](https://docs.csc.fi/support/tutorials/puhti_quick)

## Connecting to Puhti

The training user accounts are of the form `trainingXXX`, replace
`XXX` by the number of your individual account. You can then log in 
with the provided password via `ssh`:
```bash
ssh trainingXXX@puhti.csc.fi
```

For editing program source files you can use e.g. *nano* editor :

```
nano prog.f90
```
(`^` in nano's shortcuts refer to **Ctrl** key, *i.e.* in order to save file and exit editor press `Ctrl+X`).
Also other editors such as *vim* and *emacs*  are available.


## HIP environment in Puhti

Puhti uses a *module* system, HIP environment can be enabled with

```bash
module load hip/4.0.0c

module list

Currently Loaded Modules:
  1) StdEnv   2) gcc/9.1.0   3) cuda/11.1.0   4) hip/4.0.0c   5) intel-mkl/2019.0.4   6) hpcx-mpi/2.4.0
```

There is also a module _hip/4.0.0_ but we created also one _hip/4.0.0c_ which is an installation from the source code. The name will comply with the version in the future.

Current HIP configuration can be investigated with the `hipconfig`
command:
```bash
hipconfig

HIP version  : 4.0.20496-4f163c6

== hipconfig
HIP_PATH     : /appl/opt/rocm/rocm-4.0.0c/hip
ROCM_PATH    : /appl/opt/rocm/rocm-4.0.0c/
...
```

## Compiling

Source code containing HIP calls should be compiled with the `hipcc`
compiler wrapper. In NVIDIA system `hipcc`, one can use `.cu`
ending for source files, or as recommended approach use `.cpp` ending
together with the `--x cu` compiler option:
```bash
hipcc --x cu -o my_hip_program my_hip_source.cpp
```

If `hipcc` is not used in the linking phase, one needs to add
`-lcudart` linker flag:
```bash
mpicc -o my_hip_program my_hip_source.o my_c_source.o -lcudart
```

## Batch jobs

Puhti has SLURM batch queue system, a basic batch job script for
submitting a job with one GPU:

```
#!/bin/bash
#SBATCH --job-name=hip_example
#SBATCH --account=project_2000745
#SBATCH --partition=gpu
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
#SBATCH --reservation=hip101

module load hip/4.0.0c

srun my_hip_program
```

You can submit the script with

```bash
sbatch sub.sh
```

check the status of a job with

```bash
squeue -u $USER
```

and cancel a job with

```bash
scancel JOBID
```

Note: the `--reservation=hip101` SLURM option works only with the
training accounts (i.e. not with normal CSC user account) and it is
valid only during the course.
