quasi-discrete-Hankel-transform
Quasi-discrete Hankel transform for CUDA enabled gpu.

Example of how to do CUDA quasi-discret Hankel transform. The space grid is obtained from the solutions of the first kind Bessel functions j0(x)=0. The solutions must be provided in advanced in the file bessel_zeros.in. The same function is used for for doing forward or backward transform. If a forward (r->k) transform is made the results must be multiplied by rmax/kmax.. All the arrays needed for transform are kept in a structure which is iniliazed by the hankelinit function.. Only double precision is available. The flag -arch=sm_xx should have xx>=20 .

Lots of thanks to njuffa for giving very good advice on devtalk nvidia forum.
