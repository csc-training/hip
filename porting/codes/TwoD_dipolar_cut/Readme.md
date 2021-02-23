# Overdamped molecular dynamics 3D dipolar particles

This is an example of a `N-square` algorithm for particles interacting with `r^[-3]` potential. The potential is cut off at half of the simulation box. This code also contains an example of setting streams, basically performing `nstreams` copies of identical systems. This procedures allows to collect more measurements, needed for performing ensemble averages of variables of interest.  The effect of the temperature is put in as random forces drawn from normal distributions. In CUDA this is done using the `curand` library.

The [CUDA to HIP conversion](../../README.md) works, but the compilation fails  in this case due to missing `hiprand` library on puhti. To be checked later. 
Alternatevely one could remove the random numbers.
