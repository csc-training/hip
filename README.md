# HIP 101 Porting CUDA codes to HIP
### 26 February 2021


## Schedule


| Time (in CET) | Topic |
| ---- | --- |
| 09:00 - 10:00 | Introduction to AMD architecture and HIP| 
| 10:00 - 10:15 | Break | 
| 10:15 - 10:45 | Deep dive to Hipify tools and examples | 
| 10:45 - 11:30 | Lunch | 
| 11:30 - 16:00 | Hands-on sessions | 


## Important Information
Github repository: https://github.com/csc-training/hip
Submitting jobs to Puhti: https://docs.csc.fi/computing/running/submitting-jobs/
SLURM reservation for this training: _gpu_training_
AMD porting guide: https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-porting-guide.html

## Team
Instructor: George Markomanolis

Team: Cristian-Vasile Achim, Jussi Enkovaara, Fredrik Robertsen, Nicolino lo Gullo

## Structure of the repository

```bash
docs
porting
└── codes
    ├── saxpy/cuda 
    ├── saxpy/cublas     
    ├── Discrete_Hankel_Transform
    ├── Heat-Equation
    ├── 2D Wave Propagation
    ├── KMeans clustering
    └── Vector_addition

```

## Puhti
### Connect to Puhti
```bash
ssh trainingXXX@puhti.csc.fi
```
* Give your password and you should be located in the directory:
```bash=num
/users/trainingXXX
```

### Explore the environment

```bash=
 module list

Currently Loaded Modules:
  1) intel/19.0.4   2) hpcx-mpi/2.4.0   3) intel-mkl/2019.0.4   4) StdEnv
```
### SLURM

* Submit script sub.sh

```bash
sbatch sub.sh
```

* Check the status of a job

```bash
squeue -u $USER
```

* Cancel a job

```bash
scancel JOBID
```


### HIP

* Load HIP module
```bash
module load hip/4.0.0c

module list

Currently Loaded Modules:
  1) StdEnv   2) gcc/9.1.0   3) cuda/11.1.0   4) hip/4.0.0c   5) intel-mkl/2019.0.4   6) hpcx-mpi/2.4.0
```

There is also a module _hip/4.0.0_ but we created also one _hip/4.0.0c_ which is an installation from the source code. The name will comply with the version in the future.

* hipconfig
 
```bash=
hipconfig

HIP version  : 4.0.20496-4f163c6

== hipconfig
HIP_PATH     : /appl/opt/rocm/rocm-4.0.0c/hip
ROCM_PATH    : /appl/opt/rocm/rocm-4.0.0c/
HIP_COMPILER : clang
HIP_PLATFORM : nvcc
HIP_RUNTIME  : ROCclr
CPP_CONFIG   :  -D__HIP_PLATFORM_NVCC__=  -I/appl/opt/rocm/rocm-4.0.0c/hip/include -I/appl/spack/install-tree/gcc-9.1.0/cuda-11.1.0-vvfuk2//include

== nvcc
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Tue_Sep_15_19:10:02_PDT_2020
Cuda compilation tools, release 11.1, V11.1.74
Build cuda_11.1.TC455_06.29069683_0

=== Environment Variables
PATH=/appl/opt/rocm/rocm-4.0.0c/hip/bin:/appl/spack/install-tree/gcc-9.1.0/hwloc-2.0.2-wqrgpf/bin:/appl/opt/ucx/1.9.0-cuda/bin:/appl/spack/install-tree/gcc-9.1.0/openmpi-4.0.5-ym53tz/bin:/appl/spack/install-tree/gcc-9.1.0/hdf5-1.12.0-wtlera/bin:/appl/spack/install-tree/gcc-9.1.0/cuda-11.1.0-vvfuk2/bin:/appl/spack/install-tree/gcc-4.8.5/gcc-9.1.0-vpjht2/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/appl/bin:/users/markoman/.local/bin:/users/markoman/bin
CUDA_PATH=/appl/spack/install-tree/gcc-9.1.0/cuda-11.1.0-vvfuk2/
HIPFORT_ARCH=nvptx
HIP_PLATFORM=nvcc
LD_LIBRARY_PATH=/appl/opt/rocm/rocm-4.0.0c/hip/lib:/appl/spack/install-tree/gcc-9.1.0/hwloc-2.0.2-wqrgpf/lib:/appl/opt/ucx/1.9.0-cuda/lib:/appl/spack/install-tree/gcc-9.1.0/openmpi-4.0.5-ym53tz/lib:/appl/spack/install-tree/gcc-9.1.0/hdf5-1.12.0-wtlera/lib:/appl/opt/cluster_studio_xe2019/compilers_and_libraries_2019.4.243/linux/tbb/lib/intel64_lin/gcc4.7:/appl/opt/cluster_studio_xe2019/compilers_and_libraries_2019.4.243/linux/compiler/lib/intel64_lin:/appl/opt/cluster_studio_xe2019/compilers_and_libraries_2019.4.243/linux/mkl/lib/intel64_lin:/appl/spack/install-tree/gcc-9.1.0/cuda-11.1.0-vvfuk2/lib64:/appl/spack/install-tree/gcc-4.8.5/gcc-9.1.0-vpjht2/lib64:/appl/spack/install-tree/gcc-4.8.5/gcc-9.1.0-vpjht2/lib:/appl/opt/rocm/rocm-4.0.0/hiprand/lib:/appl/opt/rocm/rocm-4.0.0c/hipblas/hipblas/lib
HIP_RUNTIME=ROCclr
HIPFORT_GPU=sm_70
CUDA_INSTALL_ROOT=/appl/spack/install-tree/gcc-9.1.0/cuda-11.1.0-vvfuk2
HIPFORT_HOME=/appl/opt/rocm/rocm-4.0.0c//hipfort/
HIPFORT_ARCHGPU=nvptx-sm_70
HIPCC_OPTS=--x cu
HIP_COMPILER=clang
HIP_PATH=/appl/opt/rocm/rocm-4.0.0c/hip

== Linux Kernel
Hostname     : puhti-login1.bullx
Linux puhti-login1.bullx 3.10.0-1062.33.1.el7.x86_64 #1 SMP Thu Aug 13 10:55:03 EDT 2020 x86_64 x86_64 x86_64 GNU/Linux
LSB Version:	:core-4.1-amd64:core-4.1-noarch
Distributor ID:	RedHatEnterpriseServer
Description:	Red Hat Enterprise Linux Server release 7.7 (Maipo)
Release:	7.7
Codename:	Maipo
```

* The wrapper to compile on NVIDIA system is called _hipcc_

```bash=
 which hipcc
/appl/opt/rocm/rocm-4.0.0c/hip/bin/hipcc
```
* You can read the file _/appl/opt/rocm/rocm-4.0.0c/hip/bin/hipcc_ for more information

```bash=
hipcc -h

Usage  : nvcc [options] <inputfile>

Options for specifying the compilation phase
============================================
More exactly, this option specifies up to which stage the input files must be compiled,
according to the following compilation trajectories for different input file types:
        .c/.cc/.cpp/.cxx : preprocess, compile, link
        .o               : link
        .i/.ii           : compile, link
        .cu              : preprocess, cuda frontend, PTX assemble,
                           merge with host C code, compile, link
        .gpu             : cicc compile into cubin
        .ptx             : PTX assemble into cubin.

```

## Porting CUDA codes to HIP

### General Guidelines 

* Start porting the CUDA codes on an NVIDIA system
* When it is finished, compile the code with HIP on an AMD system (no access to AMD hardware yet)
* HIP can be used on both AMD and NVIDIA GPUs
* The script __hipconvertinplace-perl.sh__ can hipify all the files in a directory
* Some HIP libraries seem not to work on NVIDIA systems

### VERBOSE Mode

* If you want to see the command that is executed from hipcc, declare the following_

```bash
export HIPCC_VERBOSE=1
```

* For example, on Puhti, the command:
```bash
hipcc "--gpu-architecture=sm_70" -g -O3 -I../common -c core_cuda.cu -o core_cuda.o
```

would also print the command that was actually executed:

```bash
hipcc-cmd: /appl/spack/install-tree/gcc-9.1.0/cuda-11.1.0-vvfuk2//bin/nvcc -D__HIP_ROCclr__ -Wno-deprecated-gpu-targets  -isystem /appl/spack/install-tree/gcc-9.1.0/cuda-11.1.0-vvfuk2//include -isystem /appl/opt/rocm/rocm-4.0.0c/hip/include  --gpu-architecture=sm_70 -g -O3 -I../common -c core_cuda.cu -o core_cuda.o
```

### Debug hipcc

Add in your submission script before srun:

```
export AMD_LOG_LEVEL=4
```

For example, with no debug mode:

```
srun: error: r01g01: task 0: Segmentation fault
srun: Terminating job step 4339273.0
```

with debug mode:

```
:3:rocdevice.cpp            :458 : 2193024923864 us: Initializing HSA stack.
:1:rocdevice.cpp            :466 : 2193024923948 us: hsa_init failed.
:4:runtime.cpp              :82  : 2193024923950 us: init
srun: error: r01g01: task 0: Segmentation fault
srun: Terminating job step 4339273.0
```

The outcome is that the used library does require AMD hardware and it crashes immediately. In a real execution you will observe a lot of output data.


## Exercises - Demonstration

In this point, we assume that you have cloned the github repository 

Clone the Git repository of the training:

```bash
$ git clone https://github.com/csc-training/hip.git
$ cd hip
$ export rootdir=$PWD
```

Acknowledgment: Some exercises were provided by Cristian-Valise Achim, Jussi Enkovaara, AMD, and found online.

### Exercise: SAXPY CUDA
#### Steps

SAXPY is used for Single-Precision A*X Plus Y. It combines a scalar multiplication and vector addition.

```bash 
cd ${rootdir}/porting/codes/saxpy/cuda
```
##### Check the file saxpy.cu  with an editor 

```c=
#include <stdio.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
  int N = 1<<30;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}
```
##### Compile CUDA code

```bash 
make clean
make
```

##### Submit 

```bash 
sbatch sub.sh
```
Check the files out_* and error_*

The error output includes the duration for the execution which is close to 7.1 seconds and the out_* file includes the max error which should be 0.

#### Hipify

```bash 
cp Makefile saxpy.cu ../hip/
cd ../hip
```

* Examine the hipify procedure

```bash= 
module load hip/4.0.0c
hipexamine-perl.sh saxpy.cu

  info: converted 14 CUDA->HIP refs ( error:0 init:0 version:0 device:0 context:0 module:0 memory:7 virtual_memory:0 addressing:0 stream:0 event:0 external_resource_interop:0 stream_memory:0 execution:0 graph:0 occupancy:0 texture:0 surface:0 peer:0 graphics:0 profiler:0 openGL:0 D3D9:0 D3D10:0 D3D11:0 VDPAU:0 EGL:0 thread:0 complex:0 library:0 device_library:0 device_function:3 include:0 include_cuda_main_header:0 type:0 literal:0 numeric_literal:3 define:0 extern_shared:0 kernel_launch:1 )
  warn:0 LOC:42 in 'saxpy.cu'
  hipMemcpy 3
  hipFree 2
  hipMemcpyHostToDevice 2
  hipMalloc 2
  hipLaunchKernelGGL 1
  hipMemcpyDeviceToHost 1
```

There is no warning, thus all the code can be hipified.

```bash 
hipify-perl --inplace saxpy.cu 
```

The file saxpy.cu is hipified:

```c=
#include "hip/hip_runtime.h"
#include <stdio.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
  int N = 1<<30;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  hipMalloc(&d_x, N*sizeof(float));
  hipMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  hipMemcpy(d_x, x, N*sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_y, y, N*sizeof(float), hipMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
hipLaunchKernelGGL(saxpy, dim3((N+255)/256), dim3(256), 0, 0, N, 2.0f, d_x, d_y);

  hipMemcpy(y, d_y, N*sizeof(float), hipMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);

  hipFree(d_x);
  hipFree(d_y);
  free(x);
  free(y);
}
```

14 CUDA calls were converted with no errors.

##### Compile

* Edit the Makefile and change the _nvcc_ to _hipcc_

```cmake
CC        = nvcc
```
Modify to

```cmake
CC        = hipcc
```

* Clean and compile

```bash
make clean
make
```


* Submit

```bash 
sbatch sub.sh
```
Check the files out_* and error_*

The error output includes the duration for the execution which is close to 7.32 seconds and the out_* file includes the max error which should be 0. The overhead seems to be close to 3%.

The solution is here: https://github.com/csc-training/hip/tree/main/porting/codes/saxpy/hip_solution

##### If you want to chance the .cu file to .cpp

```bash
mv saxpy.cu saxpy.cpp
```

* Edit Makefile

    * HIP Makefile with .cu

```cmake=
# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
#
#===============================================================================
# Program name & source code list
#===============================================================================
program = saxpy
source = saxpy.cu
obj = $(source:.cu=.o)
#===============================================================================
# Sets Flags
#===============================================================================
# Standard Flags
CFLAGS := -Xcompiler -Wall
# Linker Flags
LDFLAGS =
# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================
#
$(program): $(obj) Makefile
        $(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu Makefile
        $(CC) $(CFLAGS) -c $< -o $@
```
    * HIP Makefile with .cpp

```cmake=
# Compiler can be set below, or via environment variable
CC        = hipcc
HIP_CU    = hipcc --x cu
OPTIMIZE  = yes
#
#===============================================================================
# Program name & source code list
#===============================================================================
program = saxpy
source = saxpy.cpp
obj = $(source:.cpp=.o)
#===============================================================================
# Sets Flags
#===============================================================================
# Standard Flags
CFLAGS := -Xcompiler -Wall
# Linker Flags
LDFLAGS =
# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================
#
$(program): $(obj) Makefile
        $(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp Makefile
        $(HIP_CU) $(CFLAGS) -c $< -o $@

```

### Lessons learned:

* Replace __nvcc__ with __hipcc__ in the Makefile
* Hipify in-place with `hipify-perl --inplace filename`
* For NVIDIA system, if the HIP code is in a file with extension __.cpp__ use __hipcc --x cu__ instead of __hipcc__

### Exercise: SAXPY CUBLAS
#### Steps

SAXPY but using cuBLAS

```bash 
cd ${rootdir}/porting/codes/saxpy/cublas
```
##### Check the file with an editor

```cpp=
#include <iostream>
#include "cublas_v2.h"
using namespace std;

int N = 1 << 30;

int main(){
        float *a_h, *b_h;
        a_h = new float[N];
        b_h = new float[N];
        float *a_d, *b_d;
        for(int i = 0; i < N; i++){
                a_h[i] = 1.0f;
                b_h[i] = 2.0f ;
        }
        cublasHandle_t handle;
        cublasCreate(&handle);
        cudaMalloc((void**) &a_d, sizeof(float) * N);
        cudaMalloc((void**) &b_d, sizeof(float) * N);
        cublasSetVector( N, sizeof(float), a_h, 1, a_d, 1);
        cublasSetVector( N, sizeof(float), b_h, 1, b_d, 1);
        const float s = 2.0f;
        cublasSaxpy( handle, N, &s, a_d, 1, b_d, 1);
        cublasGetVector( N, sizeof(float), b_d, 1, b_h, 1);
        cudaFree(a_d);
        cudaFree(b_d);
        cublasDestroy(handle);
        float maxError = 0.0f;

        for(int i = 0; i < N; i++)
                maxError = max(maxError, abs(b_h[i]-4.0f));

        cout << "Max error: " << maxError << endl;


        delete[] a_h;
        delete[] b_h;
        return 0;
}
```
##### Compile CUDA code

```bash 
make clean
make
```

##### Submit 

```bash 
sbatch sub.sh
```
Check the files out_* and error_*

The error output includes the duration for the execution which is close to 7.1 seconds and the out_* file includes the max error which should be 0.

#### Hipify
* Examine the hipify procedure

```bash= 
make clean
cp * ../hipblas
cd ../hipblas
module load hip/4.0.0
hipexamine-perl.sh saxpy_cublas.cu 
  info: converted 12 CUDA->HIP refs ( error:0 init:0 version:0 device:0 context:0 module:0 memory:4 virtual_memory:0 addressing:0 stream:0 event:0 external_resource_interop:0 stream_memory:0 execution:0 graph:0 occupancy:0 texture:0 surface:0 peer:0 graphics:0 profiler:0 openGL:0 D3D9:0 D3D10:0 D3D11:0 VDPAU:0 EGL:0 thread:0 complex:0 library:6 device_library:0 device_function:0 include:0 include_cuda_main_header:1 type:1 literal:0 numeric_literal:0 define:0 extern_shared:0 kernel_launch:0 )
  warn:0 LOC:39 in 'saxpy_cublas.cu'
  hipFree 2
  hipMalloc 2
```

We observe the there are 6 library calls that will be converted. Hipify the code:

```bash
hipify-perl --inplace saxpy_cublas.cu 
```

Now the code is:

```cpp=
#include <iostream>
#include "hipblas.h"
using namespace std;

const int N = 1 << 30;

int main(){
        float *a_h, *b_h;
        a_h = new float[N];
        b_h = new float[N];
        float *a_d, *b_d;
        for(int i = 0; i < N; i++){
                a_h[i] = 1.0f;
                b_h[i] = 2.0f ;
        }
        hipblasHandle_t handle;
        hipblasCreate(&handle);
        hipMalloc((void**) &a_d, sizeof(float) * N);
        hipMalloc((void**) &b_d, sizeof(float) * N);
        hipblasSetVector( N, sizeof(float), a_h, 1, a_d, 1);
        hipblasSetVector( N, sizeof(float), b_h, 1, b_d, 1);
        const float s = 2.0f;
        hipblasSaxpy( handle, N, &s, a_d, 1, b_d, 1);
        hipblasGetVector( N, sizeof(float), b_d, 1, b_h, 1);
        hipFree(a_d);
        hipFree(b_d);
        hipblasDestroy(handle);
        float maxError = 0.0f;

        for(int i = 0; i < N; i++)
                maxError = max(maxError, abs(b_h[i]-4.0f));

        cout << "Max error: " << maxError << endl;


        delete[] a_h;
        delete[] b_h;
        return 0;
}
```

#### Compile

* Modify the Makefile to:
```cmake=
...
CC        = hipcc
...
CFLAGS := -Xcompiler -Wall -I/appl/opt/rocm/rocm-4.0.0c/hipblas/hipblas/include
...
LDFLAGS = -L/appl/opt/rocm/rocm-4.0.0c/hipblas/hipblas/lib/ -lhipblas
...
```

* Define variables to find the hipBLAS header and library and compile

```bash=
export LD_LIBRARY_PATH=/appl/opt/rocm/rocm-4.0.0c/hipblas/hipblas/lib/:$LD_LIBRARY_PATH
```
Load the custom installation of ROCm
```bash=
module load hip/4.0.0c
make clean
make
```

* Submit your job script

```bash
sbatch sub.sh
```
* Check the out* and error* files.

The solution is here: https://github.com/csc-training/hip/tree/main/porting/codes/saxpy/hipblas_solution

### Lessons learned:

* Always link with the appropriate library when it is available
* Do not forget to declare the _LD_LIBRARY_PATH_ environment variable
* Adjust the Makefile


### Exercise: Discrete_Hankel_Transform

Description: https://github.com/csc-training/hip/tree/main/porting/codes/Discrete_Hankel_Transform

#### Steps
```bash 
cd ${rootdir}porting/codes/Discrete_Hankel_Transform/cuda
```

##### Compile and Execute
```bash
nvcc -arch=sm_70  -o code Code.cu
sbatch sub.sh
```

#### Hipify, Compile, and Execute
```bash=
cp * ../hip/
cd ../hip/

$ hipexamine-perl.sh Code.cu 
  info: converted 46 CUDA->HIP refs ( error:0 init:0 version:0 device:0 context:0 module:0 memory:24 virtual_memory:0 addressing:0 stream:0 event:0 external_resource_interop:0 stream_memory:0 execution:0 graph:0 occupancy:0 texture:0 surface:0 peer:0 graphics:0 profiler:0 openGL:0 D3D9:0 D3D10:0 D3D11:0 VDPAU:0 EGL:0 thread:0 complex:0 library:0 device_library:0 device_function:11 include:0 include_cuda_main_header:0 type:0 literal:0 numeric_literal:9 define:0 extern_shared:0 kernel_launch:2 )
  warn:0 LOC:220 in 'Code.cu'
  hipMalloc 9
  hipMemcpy 9
  hipMemcpyHostToDevice 7
  hipFree 3
  hipLaunchKernelGGL 2
  hipMemGetInfo 2
  hipMemcpyDeviceToHost 2
  hipMemset 1

$ hipify-perl -inplace Code.cu 

$ ls
bessel_zeros.in  Code.cu  Code.cu.prehip  README.md  sub.sh

hipcc -arch=sm_70  -o code Code.cu
sbatch sub.sh
```

The solution is here: https://github.com/csc-training/hip/tree/main/porting/codes/Discrete_Hankel_Transform/hip_solution

### Exercise: Heat Equation

Example implementations of two dimensional heat equation.

#### CUDA

```bash
cd ${rootdir}/porting/codes/heat-equation/cuda
```

* Load CUDA aware MPI

```bash
module load openmpi
```

* Compile and execute

```bash=
make
sbatch sub.sh
```

* Check the out* file, for example:

```bash=
cat out_5003895
Average temperature at start: 59.763305
Iteration took 0.179 seconds.
Average temperature: 59.281239
Reference value with default arguments: 59.281239

```

#### Hipify
```bash=
make clean
mkdir ../hip
cp *.cpp *.h *.cu ../hip/
cd ../hip

hipexamine-perl.sh 
  info: converted 13 CUDA->HIP refs ( error:0 init:0 version:0 device:1 context:0 module:0 memory:6 virtual_memory:0 addressing:0 stream:0 event:0 external_resource_interop:0 stream_memory:0 execution:0 graph:0 occupancy:0 texture:0 surface:0 peer:0 graphics:0 profiler:0 openGL:0 D3D9:0 D3D10:0 D3D11:0 VDPAU:0 EGL:0 thread:0 complex:0 library:0 device_library:0 device_function:0 include:1 include_cuda_main_header:0 type:0 literal:0 numeric_literal:4 define:0 extern_shared:0 kernel_launch:1 )
  warn:0 LOC:90 in './core_cuda.cu'
  info: converted 3 CUDA->HIP refs ( error:0 init:0 version:0 device:2 context:0 module:0 memory:0 virtual_memory:0 addressing:0 stream:0 event:0 external_resource_interop:0 stream_memory:0 execution:0 graph:0 occupancy:0 texture:0 surface:0 peer:0 graphics:0 profiler:0 openGL:0 D3D9:0 D3D10:0 D3D11:0 VDPAU:0 EGL:0 thread:0 complex:0 library:0 device_library:0 device_function:0 include:1 include_cuda_main_header:0 type:0 literal:0 numeric_literal:0 define:0 extern_shared:0 kernel_launch:0 )
  warn:0 LOC:200 in './setup.cpp'

  info: TOTAL-converted 16 CUDA->HIP refs ( error:0 init:0 version:0 device:3 context:0 module:0 memory:6 virtual_memory:0 addressing:0 stream:0 event:0 external_resource_interop:0 stream_memory:0 execution:0 graph:0 occupancy:0 texture:0 surface:0 peer:0 graphics:0 profiler:0 openGL:0 D3D9:0 D3D10:0 D3D11:0 VDPAU:0 EGL:0 thread:0 complex:0 library:0 device_library:0 device_function:0 include:2 include_cuda_main_header:0 type:0 literal:0 numeric_literal:4 define:0 extern_shared:0 kernel_launch:1 )
  warn:0 LOC:702
  kernels (1 total) :   evolve_kernel(1)

  hipMemcpy 4
  hipMemcpyHostToDevice 3
  hipMalloc 2
  hip_runtime_api 2
  hipGetDeviceCount 1
  hipDeviceSynchronize 1
  hipLaunchKernelGGL 1
  hipSetDevice 1
  hipMemcpyDeviceToHost 1

hipconvertinplace-perl.sh .
  info: converted 13 CUDA->HIP refs ( error:0 init:0 version:0 device:1 context:0 module:0 memory:6 virtual_memory:0 addressing:0 stream:0 event:0 external_resource_interop:0 stream_memory:0 execution:0 graph:0 occupancy:0 texture:0 surface:0 peer:0 graphics:0 profiler:0 openGL:0 D3D9:0 D3D10:0 D3D11:0 VDPAU:0 EGL:0 thread:0 complex:0 library:0 device_library:0 device_function:0 include:1 include_cuda_main_header:0 type:0 literal:0 numeric_literal:4 define:0 extern_shared:0 kernel_launch:1 )
  warn:0 LOC:90 in './core_cuda.cu'
  info: converted 3 CUDA->HIP refs ( error:0 init:0 version:0 device:2 context:0 module:0 memory:0 virtual_memory:0 addressing:0 stream:0 event:0 external_resource_interop:0 stream_memory:0 execution:0 graph:0 occupancy:0 texture:0 surface:0 peer:0 graphics:0 profiler:0 openGL:0 D3D9:0 D3D10:0 D3D11:0 VDPAU:0 EGL:0 thread:0 complex:0 library:0 device_library:0 device_function:0 include:1 include_cuda_main_header:0 type:0 literal:0 numeric_literal:0 define:0 extern_shared:0 kernel_launch:0 )
  warn:0 LOC:200 in './setup.cpp'

  info: TOTAL-converted 16 CUDA->HIP refs ( error:0 init:0 version:0 device:3 context:0 module:0 memory:6 virtual_memory:0 addressing:0 stream:0 event:0 external_resource_interop:0 stream_memory:0 execution:0 graph:0 occupancy:0 texture:0 surface:0 peer:0 graphics:0 profiler:0 openGL:0 D3D9:0 D3D10:0 D3D11:0 VDPAU:0 EGL:0 thread:0 complex:0 library:0 device_library:0 device_function:0 include:2 include_cuda_main_header:0 type:0 literal:0 numeric_literal:4 define:0 extern_shared:0 kernel_launch:1 )
  warn:0 LOC:702
  kernels (1 total) :   evolve_kernel(1)

  hipMemcpy 4
  hipMemcpyHostToDevice 3
  hipMalloc 2
  hip_runtime_api 2
  hipGetDeviceCount 1
  hipDeviceSynchronize 1
  hipLaunchKernelGGL 1
  hipSetDevice 1
  hipMemcpyDeviceToHost 1

ls
core.cpp	 core_cuda.cu	      heat.h	     io.cpp	    main.cpp	     Makefile	    setup.cpp	      sub2.sh  utilities.cpp
core.cpp.prehip  core_cuda.cu.prehip  heat.h.prehip  io.cpp.prehip  main.cpp.prehip  Makefile_orig  setup.cpp.prehip  sub.sh   utilities.cpp.prehip
```
#### Update the Makefile

* Original Makefile
```cmake=
ifeq ($(COMP),)
COMP=gnu
endif
...

ifeq ($(COMP),gnu)
CXX=mpicxx
CC=gcc
NVCC=nvcc
NVCCFLAGS=-g -O3 -I$(COMMONDIR)
CCFLAGS=-g -O3 -Wall -I$(COMMONDIR)
LDFLAGS=
LIBS=-lpng -lcudart
endif

EXE=heat_cuda
OBJS=main.o core.o core_cuda.o setup.o utilities.o io.o
OBJS_PNG=$(COMMONDIR)/pngwriter.o

...
$(EXE): $(OBJS) $(OBJS_PNG)
        $(CXX) $(CCFLAGS) $(OBJS) $(OBJS_PNG) -o $@ $(LDFLAGS) $(LIBS)

%.o: %.cpp
        $(CXX) $(CCFLAGS) -c $< -o $@

%.o: %.c
        $(CC) $(CCFLAGS) -c $< -o $@

%.o: %.cu
        $(NVCC) $(NVCCFLAGS) -c $< -o $@
```

* Tips
    * Use __hipcc__ to compile the code with HIP calls 
    * __hipcc__ can add the necessary options to link with the default libraries that are required (not the MPI etc.)

* New Makefile with regards that we use __nvcc__ under the __hipcc__

```cmake=
ifeq ($(COMP),)
COMP=hipcc
endif
...
ifeq ($(COMP),hipcc)
CXX=hipcc
CC=gcc
NVCC=hipcc --x cu
NVCCFLAGS=-g -O3 -I$(COMMONDIR)
CXXFLAGS=-g -O3 -Xcompiler -Wall -I$(COMMONDIR)
CCFLAGS=-g -O3 -Wall -I$(COMMONDIR)
LDFLAGS=
LIBS=-lpng -lmpi
endif
...
$(EXE): $(OBJS) $(OBJS_PNG)
        $(CXX) $(CXXFLAGS) $(OBJS) $(OBJS_PNG) -o $@ $(LDFLAGS) $(LIBS)

%.o: %.cpp
        $(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
        $(CC) $(CCFLAGS) -c $< -o $@

%.o: %.cu
        $(NVCC) $(NVCCFLAGS) -c $< -o $@
```

* Compile and Execute
```bash=
make
sbatch sub.sh
```

### Exercise: 2D WAVE Propagation

2D Wave Propagation

#### CUDA and CPU
The 2D Wave Propagation case was provided by Ludovic Rass

```bash=
cd ${rootdir}/porting/codes/wave_2d/cuda_cpu
ls 
compile.sh  sub.sh  vizme2D.m  Wave_2D.c  Wave_2D.cu
```

The file _Wave_2D.c_ is for CPU and the _Wave_2D.cu_ is for GPU.

* Compile and Submit
```bash=
cat compile.sh
#!/bin/bash
g++ -O3 Wave_2D.c -o wcpu

nvcc -arch=sm_70 -O3 Wave_2D.cu -o wgpu

./compile.sh

sbatch sub.sh
```
* Check the out* file

```bash=
cat out_5015029 
Perf: 220 iterations took 7.392e-03 seconds @ 32.9915 GB/s.
Process uses GPU with id 0 .
Perf: 220 iterations took 3.312e-03 seconds @ 73.6352 GB/s.
```

The CUDA code has 2.28 times better bandwidth. Of course, it depends on the problem size which in this case seems small.

##### HIP

There is a script to compile the code in the _hip_ folder already. Copy the CUDA file to the _../hip_ directory

```bash
cp *.cu sub.sh ../hip
cd ../hip
```

##### Hipify

```bash=
hipify-perl --print-stats --inplace Wave_2D.cu 
  info: converted 28 CUDA->HIP refs ( error:2 init:0 version:0 device:9 context:0 module:0 memory:4 virtual_memory:0 addressing:0 stream:0 event:0 external_resource_interop:0 stream_memory:0 execution:0 graph:0 occupancy:0 texture:0 surface:0 peer:0 graphics:0 profiler:0 openGL:0 D3D9:0 D3D10:0 D3D11:0 VDPAU:0 EGL:0 thread:0 complex:0 library:0 device_library:0 device_function:4 include:0 include_cuda_main_header:1 type:1 literal:0 numeric_literal:4 define:0 extern_shared:0 kernel_launch:3 )
  warn:0 LOC:126 in 'Wave_2D.cu'
  hipDeviceReset 3
  hipDeviceSynchronize 3
  hipLaunchKernelGGL 3
  hipMemcpy 2
  hipGetDevice 1
  hipError_t 1
  hipFree 1
  hipDeviceSetCacheConfig 1
  hipMalloc 1
  hip_runtime 1
  hipSetDevice 1
  hipMemcpyDeviceToHost 1
  hipMemcpyHostToDevice 1
  hipSuccess 1
  hipGetErrorString 1
  hipGetLastError 1
  hipFuncCachePreferL1 1
```

* Compile and submit

Before you proceed with the submission, edit the `sub.sh` and comment the srun command to execute the CPU executable

```bash=
./compile.sh
sbatch sub.sh
```

* From the output file

```bash=
cat out_*
Process uses GPU with id 0 .
Perf: 220 iterations took 3.385e-03 seconds @ 72.0481 GB/s.
```

The HIP version provides similar results to the CUDA version with a small overhead.

### Exercise: KMeans

Parallel k-means clustering code

#### CUDA

```bash=
cd ${rootdir}porting/codes/kmeans/cuda
ls
cuda_io.cu  cuda_kmeans.cu  cuda_main.cu  cuda_wtime.cu  Image_data  kmeans.h  LICENSE  Makefile  README  sample.output  sub.sh
```

* Compile and Execute

```bash=
make cuda
sbatch sub.sh
```

* We can check the out* and error* files

```bash=
Writing coordinates of K=128 cluster centers to file "Image_data/color17695.bin.cluster_centres"
Writing membership of N=17695 data objects to file "Image_data/color17695.bin.membership"

Performing **** Regular Kmeans (CUDA version) ****
Input file:     Image_data/color17695.bin
numObjs       = 17695
numCoords     = 9
numClusters   = 128
threshold     = 0.0010
Loop iterations    = 131
I/O time           =     0.0529 sec
Computation timing =     0.2059 sec
```

#### HIP

* Copy the data to the _../hip_ directory

```bash=
cp -r *.cu *.h Image_data ../hip 
cd ../hip
```

* Hipify

```bash=
hipconvertinplace-perl.sh .
  info: converted 28 CUDA->HIP refs ( error:0 init:0 version:0 device:4 context:0 module:0 memory:13 virtual_memory:0 addressing:0 stream:0 event:0 external_resource_interop:0 stream_memory:0 execution:0 graph:0 occupancy:0 texture:0 surface:0 peer:0 graphics:0 profiler:0 openGL:0 D3D9:0 D3D10:0 D3D11:0 VDPAU:0 EGL:0 thread:0 complex:0 library:0 device_library:0 device_function:1 include:0 include_cuda_main_header:0 type:1 literal:0 numeric_literal:5 define:0 extern_shared:2 kernel_launch:2 )
  warn:0 LOC:372 in './cuda_kmeans.cu'
  info: converted 8 CUDA->HIP refs ( error:3 init:0 version:0 device:0 context:0 module:0 memory:0 virtual_memory:0 addressing:0 stream:0 event:0 external_resource_interop:0 stream_memory:0 execution:0 graph:0 occupancy:0 texture:0 surface:0 peer:0 graphics:0 profiler:0 openGL:0 D3D9:0 D3D10:0 D3D11:0 VDPAU:0 EGL:0 thread:0 complex:0 library:0 device_library:0 device_function:0 include:1 include_cuda_main_header:0 type:2 literal:0 numeric_literal:1 define:1 extern_shared:0 kernel_launch:0 )
  warn:0 LOC:79 in './kmeans.h'

  info: TOTAL-converted 36 CUDA->HIP refs ( error:3 init:0 version:0 device:4 context:0 module:0 memory:13 virtual_memory:0 addressing:0 stream:0 event:0 external_resource_interop:0 stream_memory:0 execution:0 graph:0 occupancy:0 texture:0 surface:0 peer:0 graphics:0 profiler:0 openGL:0 D3D9:0 D3D10:0 D3D11:0 VDPAU:0 EGL:0 thread:0 complex:0 library:0 device_library:0 device_function:1 include:1 include_cuda_main_header:0 type:3 literal:0 numeric_literal:6 define:1 extern_shared:2 kernel_launch:2 )
  warn:0 LOC:843
  kernels (1 total) :   compute_delta(1)

  hipMemcpy 5
  hipFree 4
  hipMalloc 4
  hipMemcpyHostToDevice 3
  hipError_t 2
  hipDeviceSynchronize 2
  hipLaunchKernelGGL 2
  hipGetErrorString 2
  HIP_DYNAMIC_SHARED 2
  hipMemcpyDeviceToHost 2
  hipGetDevice 1
  hipDeviceProp_t 1
  hipGetDeviceProperties 1
  hipSuccess 1
  hipGetLastError 1
```

* Compile and execute

```bash=
make -f Makefile.hip 
sbatch sub.sh
```
* The output file

```bash=
Performing **** Regular Kmeans (CUDA version) ****
Input file:     Image_data/color17695.bin
numObjs       = 17695
numCoords     = 9
numClusters   = 128
threshold     = 0.0010
Loop iterations    = 131
I/O time           =     0.0081 sec
Computation timing =     0.2000 sec
```

### Exercise: Madgraph 4 GPU

This code developed in the context of porting the MadGraph5_aMC@NLO event generator software onto GPU hardware. MadGraph5_aMC@NLO is able to generate code for various physics processes in different programming languages (Fortran, C, C++). 

```bash=
cd ${rootdir}/porting/codes/
mkdir madgraph4gpu
cd madgraph4gpu
wget https://github.com/madgraph5/madgraph4gpu/archive/master.zip
unzip master.zip
cd madgraph4gpu-master/
ls
epoch0	epoch1	epoch2	README.md  test  tools
cd epoch1/
cp -r cuda hip
```

#### Hipify

```bash=
hipconvertinplace-perl.sh hip/
  info: converted 7 CUDA->HIP refs ( error:0 init:0 version:0 device:0 context:0 module:0 memory:3 virtual_memory:0 addressing:0 stream:0 event:0 external_resource_interop:0 stream_memory:0 execution:0 graph:0 occupancy:0 texture:0 surface:0 peer:0 graphics:0 profiler:0 openGL:0 D3D9:0 D3D10:0 D3D11:0 VDPAU:0 EGL:0 thread:0 complex:0 library:0 device_library:0 device_function:4 include:0 include_cuda_main_header:0 type:0 literal:0 numeric_literal:0 define:0 extern_shared:0 kernel_launch:0 )
  warn:0 LOC:841 in 'hip/gg_tt/SubProcesses/P1_Sigma_sm_gg_ttx/CPPProcess.cu'
...
 info: TOTAL-converted 294 CUDA->HIP refs ( error:11 init:0 version:0 device:3 context:0 module:0 memory:59 virtual_memory:0 addressing:0 stream:0 event:0 external_resource_interop:0 stream_memory:0 execution:0 graph:0 occupancy:0 texture:0 surface:0 peer:0 graphics:0 profiler:0 openGL:0 D3D9:0 D3D10:0 D3D11:0 VDPAU:0 EGL:0 thread:0 complex:14 library:6 device_library:16 device_function:35 include:0 include_cuda_main_header:1 type:26 literal:0 numeric_literal:31 define:79 extern_shared:0 kernel_launch:13 )
  warn:2 LOC:15920
  warning: unconverted cudaTearDown : 2
  kernels (2 total) :   sigmaKin(3)  gProc::sigmaKin(2)

  hipMemcpy 23
  hipMemcpyToSymbol 19
  hipMemcpyDeviceToHost 18
  hipLaunchKernelGGL 18
  hipFree 14
  hipMalloc 11
  hipPeekAtLastError 9
  hipMemcpyHostToDevice 6
  hipDeviceReset 5
  hipSuccess 5
  HIP_SYMBOL 5
  hipError_t 4
  hipGetErrorString 4
  hipDoubleComplex 3
  hip_runtime 3
  hipHostFree 3
  hipHostMalloc 3
  hipMemcpy3D 3
  hipFloatComplex 3
  hipMemcpy2D 3
  HIPRAND_STATUS_SUCCESS 2
  HIPRAND_RNG_PSEUDO_PHILOX4_32_10 2
  HIPRAND_RNG_PSEUDO_MT19937 2
  HIPRAND_RNG_PSEUDO_MTGP32 2
  HIPRAND_RNG_PSEUDO_XORWOW 2
  HIPRAND_RNG_PSEUDO_MRG32K3A 2
  hipComplex 1
  hip_complex 1
  hipCsubf 1
  hipCmulf 1
  hipCaddf 1
  hipCdiv 1
  hipCrealf 1
  hipCsub 1
  hipCreal 1
  hipCimag 1
  hipCmul 1
  hipCadd 1
  hipCimagf 1
  hipCdivf 1
```

* The warning is about a cuda variable deployed by the developers, so it is safe. The _hiprand_ is not working on our environment this moment, thus the utilization of this code requires the actual AMD hardware, however, it is ported. It requires tuning and checking.

### CUDA Fortran

* As we discussed already, there is no straight forward approach with CUDA Fortran.
* The HIP functions are callable from C and with `extern C` are callable from Fortran
* Procedure:
    * Port CUDA Fortran code to HIP kernels in C++. The hipfort helps to call some HIP calls from Fortran.
    * Wrap the kernel launch in C function
    * Call the C function from Fortran through Fortran 2003 C binding, using pointers etc.
    
### Exercise: SAXPY CUDA Fortran

We have the following example. SAXPY code in CUDA Fortran. In this case, to hipify the code, we follow this procedure.

```bash=
$ cd ${rootdir}/porting/codes/cuda_fortran_saxpy/cuda
$ ls
main.cuf
cat main.cuf
```
```fortran=
module mathOps
contains
  attributes(global) subroutine saxpy(x, y, a)
    implicit none
    real :: x(:), y(:)
    real, value :: a
    integer :: i, n
    n = size(x)
    i = blockDim%x * (blockIdx%x - 1) + threadIdx%x
    if (i <= n) y(i) = y(i) + a*x(i)
  end subroutine saxpy 
end module mathOps

program testSaxpy
  use mathOps
  use cudafor
  implicit none
  integer, parameter :: N = 40000
  real :: x(N), y(N), a
  real, device :: x_d(N), y_d(N)
  type(dim3) :: grid, tBlock

  tBlock = dim3(256,1,1)
  grid = dim3(ceiling(real(N)/tBlock%x),1,1)

  x = 1.0; y = 2.0; a = 2.0
  x_d = x
  y_d = y
  call saxpy<<<grid, tBlock>>>(x_d, y_d, a)
  y = y_d
  write(*,*) 'Max error: ', maxval(abs(y-4.0))
end program testSaxpy
```

* Compile the code and submit

```bash=
./compile.sh
sbatch sub.sh
```

* Check the out* and error* files

```bash=
 cat out_*
 Max error:     0.000000
 
 cat error*
 real	0m0.404s
 ...
```

#### Create the HIP kernel

* Original kernel

```fortran=
    i = blockDim%x * (blockIdx%x - 1) + threadIdx%x
    if (i <= n) y(i) = y(i) + a*x(i)
```

* HIP kernel

```cpp=
__global__ void saxpy(float *y, float *x, float a, int n)
{
    size_t i = blockDim.x * blockIdx.x  + threadIdx.x;
    if (i < n) y[i] = y[i] + a*x[i];
}
```

##### Observations

* \__global__ means that the function will be executed on the GPU and it will be called from the host
* In Fortran the variables such as  *blockDim%x* are used in C/C++ as *blockDim.x*. This means that you have to change all these variables but a find and replace through __sed__ could be easy
* Using arrays also is different for example `y(i)` becomes `y[i]` which again __sed__ could help
* Overall we need to be careful that we do not do any mistake, always check the results

#### Wrap the kernel launch in C function

```c=
extern "C"
{
  void launch(float **dout, float **da, float db, int N)
  {

     dim3 tBlock(256,1,1);
     dim3 grid(ceil((float)N/tBlock.x),1,1);
    
    hipLaunchKernelGGL((saxpy), grid, tBlock, 0, 0, *dout, *da, db, N);
  }
}
```
#### The new Fortran 2003 code

```fortran=
program testSaxpy
  use iso_c_binding
  use hipfort
  use hipfort_check

  implicit none
  interface
     subroutine launch(y,x,b,N) bind(c)
       use iso_c_binding
       implicit none
       type(c_ptr) :: y,x
       integer, value :: N
       real, value :: b
     end subroutine
  end interface

  type(c_ptr) :: dx = c_null_ptr
  type(c_ptr) :: dy = c_null_ptr
  integer, parameter :: N = 40000
  integer, parameter :: bytes_per_element = 4
  integer(c_size_t), parameter :: Nbytes = N*bytes_per_element
  real, allocatable,target,dimension(:) :: x, y


  real, parameter ::  a=2.0
  real :: x_d(N), y_d(N)

  call hipCheck(hipMalloc(dx,Nbytes))
  call hipCheck(hipMalloc(dy,Nbytes))

  allocate(x(N))
  allocate(y(N))

  x = 1.0;y = 2.0

  call hipCheck(hipMemcpy(dx, c_loc(x), Nbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dy, c_loc(y), Nbytes, hipMemcpyHostToDevice))

  call launch(dy, dx, a, N)

  call hipCheck(hipDeviceSynchronize())

  call hipCheck(hipMemcpy(c_loc(y), dy, Nbytes, hipMemcpyDeviceToHost))

  write(*,*) 'Max error: ', maxval(abs(y-4.0))

  call hipCheck(hipFree(dx))
  call hipCheck(hipFree(dy))

  deallocate(x)
  deallocate(y)

end program testSaxpy
```

#### Makefile 

* hipfort provides a Makefile called _Makefile.hipfort_

```cmake=
export HIPFORT_HOME=${ROCM_PATH}/hipfort/
include ${HIPFORT_HOME}/bin/Makefile.hipfort

OUTPUT_DIR ?= $(PWD)
APP         = $(OUTPUT_DIR)/saxpy

.DEFAULT_GOAL := all

all: $(APP)

$(APP): $(OUTPUT_DIR)/main.o $(OUTPUT_DIR)/hipsaxpy.o
	$(FC) $^ $(LINKOPTS) -o $(APP)

$(OUTPUT_DIR)/main.o: main.f03
	$(FC)  -c $^ -o $(OUTPUT_DIR)/main.o

$(OUTPUT_DIR)/hipsaxpy.o: hipsaxpy.cpp
	$(CXX) --x cu -c $^ -o $(OUTPUT_DIR)/hipsaxpy.o

clean:
	rm -f $(APP) *.o *.mod *~
```

__Tip:__ Not sure how safe it is but if all your cpp files had HIP calls under NVIDIA system, you could define `export HIPCC_COMPILE_FLAGS_APPEND="--x cu"` and not to modify the Makefile. Be careful as this can break something else.

* Compile and submit

```bash=
module load hip/4.0.0c
make
submit sub.sh
```

## Gromacs

GROMACS is a molecular dynamics package mainly designed for simulations of proteins, lipids, and nucleic acids.

Do not follow these instructions as it could take a long time, they are documented to help you in your case

### Download Gromacs and uncompress:
```bash=
wget https://ftp.gromacs.org/gromacs/gromacs-2021.tar.gz
tar zxvf gromacs-2021.tar.gz
cd gromacs-2021
ls
admin  api  AUTHORS  build  cmake  CMakeLists.txt  computed_checksum  COPYING  CPackInit.cmake	CTestConfig.cmake  docs  INSTALL  python_packaging  README  scripts  share  src  tests
```

### Hipify

Let's hipify the application automatically with the __hipconvertinplace-perl.sh__ script
```bash=
cd src
 hipconvertinplace-perl.sh . 
 
 info: converted 10 CUDA->HIP refs ( error:0 init:0 version:0 device:3 context:0 module:0 memory:4 virtual_memory:0 addressing:0 stream:0 event:0 external_resource_interop:0 stream_memory:0 execution:0 graph:0 occupancy:0 texture:0 surface:0 peer:0 graphics:0 profiler:0 openGL:0 D3D9:0 D3D10:0 D3D11:0 VDPAU:0 EGL:0 thread:0 complex:0 library:0 device_library:0 device_function:0 include:0 include_cuda_main_header:0 type:1 literal:0 numeric_literal:2 define:0 extern_shared:0 kernel_launch:0 )
  warn:0 LOC:89 in './gromacs/gpu_utils/tests/devicetransfers.cu'
  info: converted 13 CUDA->HIP refs ( error:0 init:0 version:0 device:0 context:0 module:0 memory:2 virtual_memory:0 addressing:0 stream:0 event:0 external_resource_interop:0 stream_memory:0 execution:0 graph:0 occupancy:0 texture:0 surface:0 peer:0 graphics:0 profiler:0 openGL:0 D3D9:0 D3D10:0 D3D11:0 VDPAU:0 EGL:0 thread:0 complex:0 library:0 device_library:0 device_function:0 include:0 include_cuda_main_header:0 type:2 literal:0 numeric_literal:8 define:1 extern_shared:0 kernel_launch:0 )
  warn:0 LOC:126 in './gromacs/gpu_utils/pinning.cu'
  info: converted 12 CUDA->HIP refs ( error:0 init:0 version:0 device:0 context:0 module:0 memory:6 virtual_memory:0 addressing:0 stream:0 event:0 external_resource_interop:0 stream_memory:0 execution:0 graph:0 occupancy:0 texture:0 surface:0 peer:0 graphics:0 profiler:0 openGL:0 D3D9:0 D3D10:0 D3D11:0 VDPAU:0 EGL:0 thread:0 complex:0 library:0 device_library:0 device_function:0 include:0 include_cuda_main_header:0 type:3 literal:0 numeric_literal:0 define:3 extern_shared:0 kernel_launch:0 )
  warn:0 LOC:113 in './gromacs/gpu_utils/pmalloc_cuda.cu'
 warn:0 LOC:113 in './gromacs/gpu_utils/pmalloc_cuda.cu'
  warning: ./gromacs/gpu_utils/gpu_utils.cu:#69 : static bool cudaProfilerRun = ((getenv("NVPROF_ID") != nullptr));
  warning: ./gromacs/gpu_utils/gpu_utils.cu:#90 :             isPinned = (memoryAttributes.type == cudaMemoryTypeHost);
  warning: ./gromacs/gpu_utils/gpu_utils.cu:#114 :     if (cudaProfilerRun)
  warning: ./gromacs/gpu_utils/gpu_utils.cu:#126 :     if (cudaProfilerRun)
  warning: ./gromacs/gpu_utils/gpu_utils.cu:#143 :     if (cudaProfilerRun)
  warning: ./gromacs/gpu_utils/gpu_utils.cu:#154 :  * \param[in] cudaCallName   name of CUDA peer access call
  warning: ./gromacs/gpu_utils/gpu_utils.cu:#160 :                                 const char*          cudaCallName)
  warning: ./gromacs/gpu_utils/gpu_utils.cu:#165 :                 gmx::formatString("%s from GPU %d to GPU %d failed", cudaCallName, gpuA, gpuB);
  warning: ./gromacs/gpu_utils/gpu_utils.cu:#175 :                         gpuA, gpuB, cudaCallName, gmx::getDeviceErrorString(stat).c_str());
  info: converted 32 CUDA->HIP refs ( error:1 init:0 version:0 device:6 context:0 module:0 memory:0 virtual_memory:0 addressing:2 stream:0 event:0 external_resource_interop:0 stream_memory:0 execution:0 graph:0 occupancy:0 texture:0 surface:0 peer:4 graphics:0 profiler:4 openGL:0 D3D9:0 D3D10:0 D3D11:0 VDPAU:0 EGL:0 thread:0 complex:0 library:0 device_library:0 device_function:0 include:1 include_cuda_main_header:0 type:7 literal:0 numeric_literal:7 define:0 extern_shared:0 kernel_launch:0 )
  warn:9 LOC:239 in './gromacs/gpu_utils/gpu_utils.cu'
  info: converted 13 CUDA->HIP refs ( error:0 init:0 version:0 device:1 context:0 module:0 memory:0 virtual_memory:0 addressing:0 stream:5 event:0 external_resource_interop:0 stream_memory:0 execution:0 graph:0 occupancy:0 texture:0 surface:0 peer:0 graphics:0 profiler:0 openGL:0 D3D9:0 D3D10:0 D3D11:0 VDPAU:0 EGL:0 thread:0 complex:0 library:0 device_library:0 device_function:0 include:0 include_cuda_main_header:0 type:4 literal:0 numeric_literal:2 define:1 extern_shared:0 kernel_launch:0 )
  warn:0 LOC:103 in './gromacs/gpu_utils/device_stream.cu'
  warning: ./gromacs/hardware/device_management.cu:#137 :     // it is enough to check for cudaErrorDevicesUnavailable only here because
  warning: ./gromacs/hardware/device_management.cu:#139 :     if (cu_err == cudaErrorDevicesUnavailable)
...
  info: converted 3 CUDA->HIP refs ( error:0 init:0 version:0 device:0 context:0 module:0 memory:0 virtual_memory:0 addressing:0 stream:0 event:0 external_resource_interop:0 stream_memory:0 execution:0 graph:0 occupancy:0 texture:0 surface:0 peer:0 graphics:0 profiler:0 openGL:0 D3D9:0 D3D10:0 D3D11:0 VDPAU:0 EGL:0 thread:0 complex:0 library:0 device_library:0 device_function:3 include:0 include_cuda_main_header:0 type:0 literal:0 numeric_literal:0 define:0 extern_shared:0 kernel_launch:0 )
  warn:0 LOC:320 in './gromacs/ewald/pme_spread.cu'
  warning: ./gromacs/ewald/pme_solve.cu:260: unsupported device function "__shfl_down_sync":         virxx += __shfl_down_sync(activeMask, virxx, 1, width);
  warning: ./gromacs/ewald/pme_solve.cu:261: unsupported device function "__shfl_up_sync":         viryy += __shfl_up_sync(activeMask, viryy, 1, width);
...
  info: converted 1 CUDA->HIP refs ( error:0 init:0 version:0 device:0 context:0 module:0 memory:0 virtual_memory:0 addressing:0 stream:0 event:0 external_resource_interop:0 stream_memory:0 execution:0 graph:0 occupancy:0 texture:0 surface:0 peer:0 graphics:0 profiler:0 openGL:0 D3D9:0 D3D10:0 D3D11:0 VDPAU:0 EGL:0 thread:0 complex:0 library:0 device_library:0 device_function:0 include:0 include_cuda_main_header:0 type:0 literal:0 numeric_literal:0 define:1 extern_shared:0 kernel_launch:0 )
  warn:0 LOC:2709 in './external/googletest/googletest/include/gtest/internal/gtest-port.h'

  info: TOTAL-converted 377 CUDA->HIP refs ( error:9 init:0 version:5 device:27 context:0 module:0 memory:29 virtual_memory:0 addressing:2 stream:18 event:27 external_resource_interop:0 stream_memory:0 execution:3 graph:0 occupancy:0 texture:3 surface:0 peer:4 graphics:0 profiler:4 openGL:0 D3D9:0 D3D10:0 D3D11:0 VDPAU:0 EGL:0 thread:0 complex:0 library:18 device_library:0 device_function:38 include:1 include_cuda_main_header:5 type:81 literal:0 numeric_literal:85 define:11 extern_shared:7 kernel_launch:0 )
  warn:45 LOC:869044
  warning: unconverted cudaProfilerRun : 4
  warning: unconverted cudaCallName : 4
  warning: unconverted cudaErrorDevicesUnavailable : 2
  warning: unconverted cudaMemoryTypeHost : 1
  kernels (0 total) : 

  hipError_t 47
  hipSuccess 46
  hipEventDestroy 8
  hipMemcpyAsync 8
  hipSetDevice 8
  HIP_DYNAMIC_SHARED 7
  hipFuncSetCacheConfig 6
  hipErrorInvalidValue 5
  ...
  hipStream_t 4
  hipHostMalloc 4
  hipGetDevice 4
  hipStreamQuery 4
  hip_runtime 4
  hipStreamSynchronize 4
  ...  
  hipHostMallocDefault 2
  hipProfilerStop 2
  hipPointerAttribute_t 2
  hipDeviceEnablePeerAccess 2
  ...
  hipDeviceGetStreamPriorityRange 1
  hipGetErrorName 1
  hipPeekAtLastError 1
  hipErrorInvalidDeviceFunction 1
  HIPFFT_SUCCESS 1
```
#### Issues


1. __Description:__ We should check all the warnings as these declare that something was not translated, it does not mean for sure that this is a mistake. If a developer declares a variable called cudaXXX the tool will report a warning but it is expected. 

```bash=
warning: ./gromacs/gpu_utils/gpu_utils.cu:#69 : static bool cudaProfilerRun = ((getenv("NVPROF_ID") != nullptr));
warning: ./gromacs/gpu_utils/gpu_utils.cu:#90 :             isPinned = (memoryAttributes.type == cudaMemoryTypeHost);
warning: ./gromacs/gpu_utils/gpu_utils.cu:#114 :     if (cudaProfilerRun)
warning: ./gromacs/gpu_utils/gpu_utils.cu:#126 :     if (cudaProfilerRun)
warning: ./gromacs/gpu_utils/gpu_utils.cu:#143 :     if (cudaProfilerRun)
warning: ./gromacs/gpu_utils/gpu_utils.cu:#154 :  * \param[in] cudaCallName   name of CUDA peer access call
warning: ./gromacs/gpu_utils/gpu_utils.cu:#160 :                                 const char*          cudaCallName)
warning: ./gromacs/gpu_utils/gpu_utils.cu:#165 :                 gmx::formatString("%s from GPU %d to GPU %d failed", cudaCallName, gpuA, gpuB);
warning: ./gromacs/gpu_utils/gpu_utils.cu:#175 :                         gpuA, gpuB, cudaCallName, gmx::getDeviceErrorString(stat).c_str());
```

```bash=
warning: ./gromacs/hardware/device_management.cu:#137 :     // it is enough to check for cudaErrorDevicesUnavailable only here because
warning: ./gromacs/hardware/device_management.cu:#139 :     if (cu_err == cudaErrorDevicesUnavailable)
```

```bash=
warn:45 LOC:869044
warning: unconverted cudaProfilerRun : 4
warning: unconverted cudaCallName : 4
warning: unconverted cudaErrorDevicesUnavailable : 2
warning: unconverted cudaMemoryTypeHost : 1
```

#### Solution:

Check the files with the warnings:
* For example the warning of cudaProfilerRun is not actually a serious issue as it is a variable declared by the developers
```bash
static bool cudaProfilerRun
```
* Similar for
```bash
 const char* cudaCallName
```
* About _cudaErrorDevicesUnavailable_ we can see from the web page https://rocmdocs.amd.com/en/latest/Programming_Guides/CUDAAPIHIPTEXTURE.html that there is no specific HIP call. 

```bash=
    if (cu_err == cudaErrorDevicesUnavailable)
    {
        return DeviceStatus::Unavailable;
    }

```
However, if we try to track the call path from the code, we can see:

```bash=

cu_err == cudaErrorDevicesUnavailable)
└── cu_err = checkCompiledTargetCompatibility(deviceInfo.id, deviceInfo.prop);
    └── static cudaError_t checkCompiledTargetCompatibility(int deviceId, const cudaDeviceProp& deviceProp)
        {
         cudaFuncAttributes attributes;
         cudaError_t        stat = cudaFuncGetAttributes(&attributes, dummy_kernel);
         ...
         return stat;
         }

```
Thus, the returned value is from the call to __cudaFuncAttributes__. From the PDF of HIP API which is more updated from the web site, we can see for example, for v4.0, see here: https://github.com/RadeonOpenCompute/ROCm/blob/master/HIP-API_Guide_v4.0.pdf where we can find that

```bash=
4.8.2.11 hipFuncGetAttributes()

Returns
hipSuccess, hipErrorInvalidValue, hipErrorInvalidDeviceFunction
```

Thus we can replace in the file _.gromacs/hardware/device_management.cu_ manually the __cudaErrorDevicesUnavailable__ with __hipErrorInvalidDeviceFunction__

* About the warning:

```bash
warning: unconverted cudaMemoryTypeHost : 1
```
We can see from the HIP programming guide PDF that there is __hipMemoryTypeHost__ and we are not sure why it was not converted, it could be a bug, so we manually do the appropriate modification in the file ./gromacs/gpu_utils/gpu_utils.cu

2. __Description:__ The Warp-Level primitives are not supported by HIP (yet)
```bash 
warning: ./gromacs/ewald/pme_solve.cu:260: unsupported device function "__shfl_down_sync":         virxx += __shfl_down_sync(activeMask, virxx, 1, width);
```
Github issue: https://github.com/ROCm-Developer-Tools/HIP/issues/1491

__Solution:__

Change the calls of __shfl\_*\_sync__ to __shfl\_*__ for example __shfl_down_sync to __shfl_down

3. __Description:__ Gromacs uses FFT

__Solution:__

The hipFFT seems not to be able to compile on NVIDIA systems, it is possible with the hip/4.0.0 but the explanation is different.

4. __Description:__

Error: _src/gromacs/utility/cuda_version_information.cu(49): error: identifier "hipDriverGetVersion" is undefined_

Code:
```cpp=
#include "gmxpre.h"
#include "cuda_version_information.h"
#include "gromacs/utility/stringutil.h"

namespace gmx
{

std::string getCudaDriverVersionString()
{
    int cuda_driver = 0;
    if (hipDriverGetVersion(&cuda_driver) != hipSuccess)
    {
        return "N/A";
    }
    return formatString("%d.%d", cuda_driver / 1000, cuda_driver % 100);
}

std::string getCudaRuntimeVersionString()
{
    int cuda_runtime = 0;
    if (hipRuntimeGetVersion(&cuda_runtime) != hipSuccess)
    {
        return "N/A";
    }
    return formatString("%d.%d", cuda_runtime / 1000, cuda_runtime % 100);
}

} // namespace gmx

```
__Solution:__

Add `#include "hip/hip_runtime.h"`

##### Compilation

This is an example it does not mean that this is the best way
```bash=
CXX=/appl/opt/rocm/rocm-4.0.0/hip/bin/hipcc cmake -DGMX_GPU=CUDA ..
make
```

## CMAKE

One cmake file, called `cmake/gmxManageNvccConfig.cmake` from Gromacs is the following:

```cmake=
#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright (c) 2012,2013,2014,2015,2016 by the GROMACS development team.
# Copyright (c) 2017,2018,2019,2020, by the GROMACS development team, led by
# Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
# and including many others, as listed in the AUTHORS file in the
# top-level source directory and at http://www.gromacs.org.
...
# set up host compiler and its options
if(CUDA_HOST_COMPILER_CHANGED)
    set(CUDA_HOST_COMPILER_OPTIONS "")

    if(APPLE AND CMAKE_C_COMPILER_ID MATCHES "GNU")
        # Some versions of gcc-4.8 and gcc-4.9 have produced errors
        # (in particular on OS X) if we do not use
        # -D__STRICT_ANSI__. It is harmless, so we might as well add
        # it for all versions.
        list(APPEND CUDA_HOST_COMPILER_OPTIONS "-D__STRICT_ANSI__")
    endif()

    work_around_glibc_2_23()

    set(CUDA_HOST_COMPILER_OPTIONS "${CUDA_HOST_COMPILER_OPTIONS}"
        CACHE STRING "Options for nvcc host compiler (do not edit!).")

    mark_as_advanced(CUDA_HOST_COMPILER CUDA_HOST_COMPILER_OPTIONS)
endif()

if (GMX_CUDA_TARGET_SM OR GMX_CUDA_TARGET_COMPUTE)
    set(GMX_CUDA_NVCC_GENCODE_FLAGS)
    set(_target_sm_list ${GMX_CUDA_TARGET_SM})
    foreach(_target ${_target_sm_list})
        list(APPEND GMX_CUDA_NVCC_GENCODE_FLAGS "-gencode;arch=compute_${_target},code=sm_${_target}")
    endforeach()
    set(_target_compute_list ${GMX_CUDA_TARGET_COMPUTE})
    foreach(_target ${_target_compute_list})
        list(APPEND GMX_CUDA_NVCC_GENCODE_FLAGS "-gencode;arch=compute_${_target},code=compute_${_target}")
    endforeach()
else()

  if(CUDA_VERSION VERSION_LESS "11.0")
        list (APPEND GMX_CUDA_NVCC_GENCODE_FLAGS "-gencode;arch=compute_30,code=sm_30")
    endif()
    list (APPEND GMX_CUDA_NVCC_GENCODE_FLAGS "-gencode;arch=compute_35,code=sm_35")
...
    list (APPEND GMX_CUDA_NVCC_GENCODE_FLAGS "-gencode;arch=compute_70,code=compute_70")
    if(NOT CUDA_VERSION VERSION_LESS "10.0")
        list (APPEND GMX_CUDA_NVCC_GENCODE_FLAGS "-gencode;arch=compute_75,code=compute_75")
    endif()
    if(NOT CUDA_VERSION VERSION_LESS "11.0")
        list (APPEND GMX_CUDA_NVCC_GENCODE_FLAGS "-gencode;arch=compute_80,code=compute_80")
    endif()
endif()

if((_cuda_nvcc_executable_or_flags_changed OR CUDA_HOST_COMPILER_CHANGED OR NOT GMX_NVCC_WORKS) AND NOT WIN32)
    message(STATUS "Check for working NVCC/C++ compiler combination with nvcc '${CUDA_NVCC_EXECUTABLE}'")
    execute_process(COMMAND ${CUDA_NVCC_EXECUTABLE} -ccbin ${CUDA_HOST_COMPILER} -c ${CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS_${_build_type}} ${CMAKE_SOURCE_DIR}/cmake/TestCUDA.cu
        RESULT_VARIABLE _cuda_test_res
        OUTPUT_VARIABLE _cuda_test_out
        ERROR_VARIABLE  _cuda_test_err
        OUTPUT_STRIP_TRAILING_WHITESPACE)

...
endif() # GMX_CHECK_NVCC

macro(GMX_SET_CUDA_NVCC_FLAGS)
    set(CUDA_NVCC_FLAGS "${GMX_CUDA_NVCC_FLAGS};${CUDA_NVCC_FLAGS}")
endmacro()


function(gmx_cuda_add_library TARGET)
    add_definitions(-DHAVE_CONFIG_H)
    # Source files generated by NVCC can include gmxmpi.h, and so
    # need access to thread-MPI.
    include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/src/external/thread_mpi/include)
    # Source files can also contain topology related files and need access to
    # the remaining external headers
    include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/src/external)

    # Now add all the compilation options
    gmx_cuda_target_compile_options(CUDA_${TARGET}_CXXFLAGS)
    list(APPEND CMAKE_CXX_FLAGS ${CUDA_${TARGET}_CXXFLAGS})
    foreach(build_type ${build_types_with_explicit_flags})
        list(APPEND CMAKE_CXX_FLAGS_${build_type} ${CUDA_${TARGET}_CXXFLAGS_${build_type}})
    endforeach()

    cuda_add_library(${TARGET} ${ARGN})
endfunction()

```

### Hipify

```bash=
hipify-cmakefile cmake/gmxManageNvccConfig.cmake > cmake/gmxManageHipConfig.cmake
 warning: cmake/gmxManageNvccConfig.cmake:#38 : unsupported macro/option : # - use the CUDA_HOST_COMPILER if defined by the user, otherwise
 warning: cmake/gmxManageNvccConfig.cmake:#39 : unsupported macro/option : # - check if nvcc works with CUDA_HOST_COMPILER and the generated nvcc and C++ flags
 warning: cmake/gmxManageNvccConfig.cmake:#42 : unsupported macro/option : #   * CUDA_HOST_COMPILER_OPTIONS    - the full host-compiler related option list passed to nvcc
 warning: cmake/gmxManageNvccConfig.cmake:#44 : unsupported macro/option : # Note that from CMake 2.8.10 FindCUDA defines CUDA_HOST_COMPILER internally,
 warning: cmake/gmxManageNvccConfig.cmake:#59 : unsupported macro/option :         list(APPEND CUDA_HOST_COMPILER_OPTIONS "-D_FORCE_INLINES")
 warning: cmake/gmxManageNvccConfig.cmake:#60 : unsupported macro/option :         set(CUDA_HOST_COMPILER_OPTIONS ${CUDA_HOST_COMPILER_OPTIONS} PARENT_SCOPE)
 warning: cmake/gmxManageNvccConfig.cmake:#64 : unsupported macro/option : gmx_check_if_changed(CUDA_HOST_COMPILER_CHANGED CUDA_HOST_COMPILER)
 warning: cmake/gmxManageNvccConfig.cmake:#67 : unsupported macro/option : if(CUDA_HOST_COMPILER_CHANGED)
 warning: cmake/gmxManageNvccConfig.cmake:#68 : unsupported macro/option :     set(CUDA_HOST_COMPILER_OPTIONS "")
 warning: cmake/gmxManageNvccConfig.cmake:#75 : unsupported macro/option :         list(APPEND CUDA_HOST_COMPILER_OPTIONS "-D__STRICT_ANSI__")
 warning: cmake/gmxManageNvccConfig.cmake:#80 : unsupported macro/option :     set(CUDA_HOST_COMPILER_OPTIONS "${CUDA_HOST_COMPILER_OPTIONS}"
 warning: cmake/gmxManageNvccConfig.cmake:#83 : unsupported macro/option :     mark_as_advanced(CUDA_HOST_COMPILER CUDA_HOST_COMPILER_OPTIONS)
 warning: cmake/gmxManageNvccConfig.cmake:#178 : unsupported macro/option : list(APPEND GMX_CUDA_NVCC_FLAGS "${CUDA_HOST_COMPILER_OPTIONS}")
 warning: cmake/gmxManageNvccConfig.cmake:#210 : unsupported macro/option : if((_cuda_nvcc_executable_or_flags_changed OR CUDA_HOST_COMPILER_CHANGED OR NOT GMX_NVCC_WORKS) AND NOT WIN32)
 warning: cmake/gmxManageNvccConfig.cmake:#212 : unsupported macro/option :     execute_process(COMMAND ${CUDA_NVCC_EXECUTABLE} -ccbin ${CUDA_HOST_COMPILER} -c ${HIP_NVCC_FLAGS} ${HIP_NVCC_FLAGS_${_build_type}} ${CMAKE_SOURCE_DIR}/cmake/TestCUDA.cu
```

The new cmake looks like:

```cmake=
 if(HIP_VERSION VERSION_LESS "11.0")
        list (APPEND GMX_CUDA_NVCC_GENCODE_FLAGS "-gencode;arch=compute_30,code=sm_30")
    endif()
    list (APPEND GMX_CUDA_NVCC_GENCODE_FLAGS "-gencode;arch=compute_35,code=sm_35")
 ...
    if(NOT HIP_VERSION VERSION_LESS "11.0")
        # Requesting sm or compute 35, 37, or 50 triggers deprecation messages with
        # nvcc 11.0, which we need to suppress for use in CI
        list (APPEND GMX_CUDA_NVCC_GENCODE_FLAGS "-Wno-deprecated-gpu-targets")
    endif()    
    ...
    macro(GMX_SET_CUDA_NVCC_FLAGS)
    set(HIP_NVCC_FLAGS "${GMX_CUDA_NVCC_FLAGS};${HIP_NVCC_FLAGS}")
endmacro()
```

* The tool will be improved and also probably you can add more variables using the file: /appl/opt/rocm/rocm-4.0.0c/hip/bin/hipify-cmakefile

Also there will be many improvements regarding CMake https://github.com/ROCm-Developer-Tools/HIP/issues/2158#issuecomment-737222202 

## Exercises

### Vector Addition

* Hipify the code in the repository: https://github.com/csc-training/hip/tree/main/porting/codes/Vector_Addition


## Known issues

- Some HIP libraries need dependencies not available on NVIDIA platform, need to investigate.

- If your CUDA kernel, includes the dim3() call, then hipify will convert wrongly. [Issue in Github](https://github.com/ROCm-Developer-Tools/HIPIFY/issues/246) . It was fixed on February 24th, not yet installed on Puhti

- In CUDA, `__CUDACC__` is defined by `nvcc`, but the HIP equivalent `__HIPCC__` is defined in `hip_runtime.h`. Thus, if code uses `__CUDACC__` without `#include <cuda_runtime_api.h>`, one needs to add manually `#include <hip_runtime.h>` to have the automatically converted `__HIPCC__` to get defined. [Issue in Github](https://github.com/ROCm-Developer-Tools/HIP/issues/29)

### Feedback:

