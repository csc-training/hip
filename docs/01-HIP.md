---
title:  HIP101 Porting CUDA codes to HIP
author: CSC - IT Center for Science
date:   2021-02
lang:   en
---

# Agenda (Times are in CET)

09:00 - 10:00 Introduction to AMD architecture and HIP  
10:00 - 10:15 Break  
10:15 - 10:45 Deep dive to Hipify tools and examples  
10:45 - 11:30 Lunch  
11:30 - 16:00 Hands-on sessions  

# Disclaimer
* AMD ecosystem is under ehave development
* All the experiments took place on NVIDIA V100 GPU (Puhti supercomputer)

# Motivation/Challenges
* LUMI will have AMD GPUs
* Need to learn how to port codes on AMD ecosystem
* Not yet access to AMD GPUs

# LUMI
![](img/lumi.png){ .center width=77% }

# AMD GPUs (Mi100 example)
![](img/amd_architecture.png){ .center width=77% }

# Differences between HIP and CUDA
* AMD GCN hardware wavefronts size is 64 (warp on CUDA is 32)
* Some CUDA library functions do not have AMD equivalents
* Shared memory and registers per thread can differ between AMD and NVIDIA hardware

# ROCm
![](img/rocm.png){ .center width=67% }

# ROCm Installation
* Many components need to be installed
* Rocm-cmake
* HSA Runtime API 
* ROCm LLVM/Clang
* ROCminfo (only for AMD HW)
* ROCM-Device-Libs
* ROCm-CompilerSupport
* ROCclr - Radeon Open Compute Common Language Runtime

# Introduction to HIP

* HIP: Heterogeneous Interface for Portability is developed by AMD to program on AMD GPUs
* It is a C++ runtime API and it supports both AMD and NVIDIA platforms
* HIP is similar to CUDA and there is no performance overhead on NVIDIA GPUs
* Many well-known libraries have been ported on HIP
* New projects or porting from CUDA, could be developed directly in HIP
* In some cases it is required to use AMD hardware for porting

# HIP Portability
 On a system with NVIDIA GPUs the hipcc, which is a compiler driver, will call the nvcc and not the hcc, as also a hip runtime will be included in the headers and it will be executed on NVIDIA GPU.
![](img/hip_portability.png){ .center width=43% }


# Differences between CUDA and HIP API
![](img/differences_cuda_hip_api.png){ .center width=87% }

# Differences between CUDA and HIP Launch Kernels
![](img/diff_kernels.png){ .center width=87% }

# HIP Terminology
|Term|Description|
|----|-----------|
|host|Executes the HIP API and can initiate kernel launches|
|default device|Each host maintains a default device. |
|active host thread|Thread running HIP API|
|HIP-Clang|Heterogeneous AMDGPU compiler 
|Hipify tools|Tools to convert CUDA code to HIP|
|hipconfig|Tool to report various configuration properties|

# HIP API

<div class="column">
* <font size="5"> Device management:</font>  
    * <font size="5">hipSetDevice(), hipGetDevice(), hipGetDeviceProperties()</font>  
- <font size="5"> Memory Management:</font>
    - <font size="5"> hipMalloc(), hipMemcpy(), hipMemcpyAsync(), hipFree()</font>

- <font size="5">Streams:</font>
    - <font size="5">hipStreamCreate(), hipSynchronize(), hipStreamSynchronize(), hipStreamFree()</font>

- <font size="5">Events:</font>
    - <font size="5">hipEventCreate(), hipEventRecord(), hipStreamWaitEvent(), hipEventElapsedTime()</font>
</div>

<div class="column">

- <font size="5">Device Kernels:</font>
    - <font size="5">\__globa__, \__device__, hipLaunchKernelGGL()</font>

- <font size="5">Device code:</font>
    - <font size="5">threadIdx, blockIdx, blockDim, \__shared__</font>
    - <font size="5">Hundreds math functions covering entire CUDA math library</font>

- <font size="5">Error handling:</font>
    - <font size="5">hipGetLastError(), hipGetErrorString()</font>
</div>


