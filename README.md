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


## Important links
Github repository: https://github.com/csc-training/hip



```bash
docs
porting
└── codes
    ├── saxpy/cuda 
    ├── saxpy/cublas     
    ├── Discrete_Hankel_Transform
    ├── TwoD_dipolar_cut
    └── 
```

## Puhti
### Connect to Puhti
```bash
ssh trainingXXX@puhti.csc.fi
```
Give your password and you should be located in the directory:
```bash=num
/users/trainingXXX
```
### Load the appropriate modules

```bash
module load hip/4.0.0c
```

### Explore the environment

```bash=
module list

Currently Loaded Modules:
  1) StdEnv   2) gcc/9.1.0   3) cuda/11.1.0   4) hip/4.0.0c   5) intel-mkl/2019.0.4   6) hpcx-mpi/2.4.0
```
## Porting CUDA codes to HIP

### General Guidelines 

* Start porting the CUDA codes on an NVIDIA system
* When it is finished, compile the code with HIP on an AMD system (no access to AMD hardware yet)
* HIP can be used on both AMD and NVIDIA GPUs
* The script __hipconvertinplace-perl.sh__ can hipify all the files in a directory

## Exercises - Demonstration

In this point we assume that you have clone the github repository 

Clone the Git repository of the training:

```bash
$ git clone https://github.com/csc-training/hip.git
$ cd hip
$ export rootdir=$PWD
```

### Exercise: SAXPY CUDA
#### Steps


```bash 
cd ${rootdir}/porting/codes/saxpy/cuda
```
##### Check the file with an editor 

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
cp Makefle saxpy.cu ../hip/
cd ../hip
```

* Examine the hipify procedure

```bash= 
module load hip/4.0.0
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

### Lessons learned:

* Replace __nvcc__ with __hipcc__ in the Makefile
* Hipify in-place with `hipify-perl --inplace filename`
* For NVIDIA system, if the HIP code is in a file with extension __.cpp__ use __hipcc --x cu__ instead of __hipcc__

### Exercise: SAXPY CUBLAS
#### Steps


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
*  
The solution is here: https://github.com/csc-training/hip/tree/main/porting/codes/saxpy/hipblas_solution

### Lessons learned:

* Always link with the appropriate library when it is available
* Do not forget to declare the _LD_LIBRARY_PATH_ environment variable
* Adjust the Makefile


### Exercise: Discrete_Hankel_Transform
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

#### HIP

##### Hipify
```bash=
make clean
mkdir ../hip
cp * ../hip/
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
##### Update the Makefile

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

### Exercise: SAXPY CUDA Fortran

## Gromacs

Do not follow these instructions as it could take long time, they are documented to help you in your case

### Download Gromacs and uncompress:
```bash=
wget https://ftp.gromacs.org/gromacs/gromacs-2021.tar.gz
tar zxvf gromacs-2021.tar.gz
cd gromacs-2021
ls
admin  api  AUTHORS  build  cmake  CMakeLists.txt  computed_checksum  COPYING  CPackInit.cmake	CTestConfig.cmake  docs  INSTALL  python_packaging  README  scripts  share  src  tests
```

### Hipify

Let hipify the application automatically with the __hipconvertinplace-perl.sh__ script
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
* For example the warning of cudaProfilerRun is not actual a serious issue as it is a vairable declared by the developers
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



## Exercises

### Vector Addition

* Hipify the code in the repository: https://github.com/csc-training/hip/tree/main/porting/codes/Vector_Addition

### Heat Equation

* Hipify the code in the repository: https://github.com/csc-training/hip/tree/main/porting/codes/heat-equation


### Feedback:

