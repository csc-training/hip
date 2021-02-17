/************************************************************************
* Description:								*
*************************************************************************
* CUDA BD-simulation of parallel dipoles in 2D with cutoff		*
* NBODY									*
* Written by Christian Vasile Achim					*
* Comments & HDF5 IO by Tobias Horn					*
*									*
*	Last change: 20.11.2013						*
************************************************************************/

/*
module load cuda
module load hdf5
nvcc -lcurand -lhdf5 code_with_streams.cu
./a.out -i
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <math.h>
#include <time.h>
#include "hdf5.h"

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
printf("Error at %s:%d\n",__FILE__,__LINE__); \
return EXIT_FAILURE;}} while(0)
	/* debug workaround: programm will interrupt if a CUDA-command (like i.e. memory allocation on device) could not be
	* performed successfully, the line of the command is stated --> this is not always conclusive. Sometimes the programm may interrupt
	* at a given line because of an error that occurred earlier. */


// #define tpb 320 //multiple of 32! tpb should be equal or slightly larger than Np/(total numbers of cells) in order to have 1 thread for 1 particle
// #define bpg 400 // blocks  per grid
// #define npc tpb

#define tpb 256 // tpb should be equal or slightly larger than Np/(total numbers of cells) in order to have 1 thread for 1 particle
#define bpg 512 // blocks  per grid
// #define npc 550 //npc should be bigger than tpb to allow for flcutuations in the number of particles, but a small number is preferred

#define PARTNUMBER tpb*bpg 		/* the maximum number of particles for this simulation */
// some performance tweaking can be done for tpb, lcell (and lcelly)
// for max performance it should be 1 thread one particle

/* HDF5 Dataset dimensions */
#define ADDSPACE 2			/* additional space in dataset used to store observables (time, energy...) */
#define MAXPART PARTNUMBER + ADDSPACE	/* number of particles + addspace stored in dataset */
#define MAXDIMS 2	/* the number of spatial coordinates per particle */

#define MAXPRM 12	/* the number of adjustable system parameters */
#define STRLEN 20	/* the maximum string length for parameter description */

#define VERSION 0.5		/* the version number of the program */

#define PI 3.14159265358979323
#define sqrtPI 1.772453851

// #define H5FORMAT H5T_NATIVE_DOUBLE
#define H5FORMAT H5T_NATIVE_FLOAT

#define OUTPATH "."

#define nstreams 32
#define minBlocksPerMultiprocessor 8 // change this so that for -arch=sm_20 is 1536/tpb, while for -arch=sm_30(35) it is 2048/tpb

// histogram
#define MAXBIN 100000

#include <stdint.h>

typedef float dat;		/* datatype for scalar values */
typedef float2 dat2;		/* datatype for 2D values */
/* if this is changed to double, the H5FORMAT has to be changed to H5T_NATIVE_DOUBLE and the curanrnd_normal in forces_shrange has to be changed to curand_nromal2 */

typedef char NAME[STRLEN];			/* define datatype NAME, which is a string for parameter description */

double PRM[MAXPRM];				/* GLOBAL VARIABLE Parameter set - known and editable by all functions */
NAME   PRMNAME[MAXPRM];				/* GLOBAL VARIABLE Parameter description */

int GO;


char FILENAME[nstreams][150];			/* GLOBAL filename */
char FILENAME2[nstreams][150];		/* alternative filename for copy-paste process (stream 2) */

/* Mersenne Twister Period parameters */
#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UPPER_MASK 0x80000000UL /* most significant w-r bits */
#define LOWER_MASK 0x7fffffffUL /* least significant r bits */

static unsigned long mt[N]; /* the array for the state vector  */
static int mti=N+1; /* mti==N+1 means mt[N] is not initialized */


/*************************************************************************************************
* Mersenne Twister Random Generator to initialize seeds(!) for CUDA pseudo random generator
*************************************************************************************************/
/* initializes mt[N] with a seed */
void init_genrand(unsigned long s)
{
    mt[0]= s & 0xffffffffUL;
    for (mti=1; mti<N; mti++) {
        mt[mti] =
	    (1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti);
        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array mt[].                        */
        /* 2002/01/09 modified by Makoto Matsumoto             */
        mt[mti] &= 0xffffffffUL;
        /* for >32 bit machines */
    }
}
/* generates a random number on [0,0xffffffff]-interval */
unsigned long genrand_int32(void)
{
    unsigned long y;
    static unsigned long mag01[2]={0x0UL, MATRIX_A};
    /* mag01[x] = x * MATRIX_A  for x=0,1 */

    if (mti >= N) { /* generate N words at one time */
        int kk;

        if (mti == N+1)   /* if init_genrand() has not been called, */
            init_genrand(5489UL); /* a default initial seed is used */

        for (kk=0;kk<N-M;kk++) {
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (;kk<N-1;kk++) {
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (mt[N-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
        mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];

        mti = 0;
    }

    y = mt[mti++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}
/*************************************************************************************************
*************************************************************************************************/
void init_histogram(const double xmin, const double xmax, const int nbins, double * bin_x)
{
  /* initializes hisogramm x-values ranging from xmin to xmax in the array bin_x. The number of bins is nbins */
  double f;
  int    i;
  f = (xmax - xmin) / (double)nbins; /* interval width between to bins */
  bin_x[0]  = xmin + 0.5*f;
  if (nbins > MAXBIN) {fprintf(stderr,"Too many bins, cannot initialise hist_x! Increase MAXBIN!\n"); exit(1); }
  for (i = 1; i < nbins; i++) {
     /* bin_x[i] is the x-position where the bin-point [i] will appear in a plot.
      * By shifting bin_x[0] by half a bin-separation, the data points
      * will appear in the middle of each interval */
     bin_x[i]  = bin_x[i-1] + f;
  }
}
/*************************************************************************************************/
int bin(const double x, const double xmin, const double xmax, const int nbins)
{
  /* returns the bin index for value x in the histogram specified by xmin,xmax,nbins */

  double fac = (double)nbins/(xmax-xmin);
  /* number of bins per length */
  if ((x >= xmin)&& (x < xmax)) {
      return (int)((x - xmin) * fac);
  }
  else {
    return -1;
  }
}
/*************************************************************************************************/
void display_prm(double *prm, NAME *prmname, char message[50])
{
	/* display all the parameters of the current dataset on screen */
	int i;
	fprintf(stderr,"*** %s ***\n",message);
 	for (i = 0; i < MAXPRM; i++){

			fprintf(stderr,"%lf\t%s\n",prm[i],prmname[i]);
	}
}
/*************************************************************************************************/
__device__ double atomicdoubleAdd(double* address, double val)
{
  unsigned long long int* address_as_ull =(unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do
  {
  	assumed = old;
	old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val +__longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
/*************************************************************************************************/



__device__ double2 dipolarInteraction(dat2 bi, dat2 bj, double2 ai,dat rcut,dat vpcutbd,dat lx,dat ly, const dat mi, const dat mj, const dat invlx, const dat invly,
				      const dat delrxA, const dat delrxB)
{
/************************************************************************
* Description:								*
*************************************************************************
* Calculates acceleration on particle i caused by the interaction	*
* with particle j. The distance between particles i and j is reduced 	*
* according to the image convention. 					*
*									*
* Written by CVA, edited by TH						*
*************************************************************************
* Arguments: 								*
*************************************************************************
* bi (bj): 2D position of particle i (j)				*
* ai: 2D acceleration vector of particle i from this interaction	*
* rcut: cutoff distance							*
* vpcutbd: negative potential gradient at cutoff (i.e. force at cutoff)	*
* lx(ly): box x(y)-dimension in native units				*
* timestep: the current (total) simulation time in native units		*
* shear_rate: the shear rate gamma 					*
************************************************************************/

// 	    int overlap=(jp-ip==0?0:1);

	    dat delrx;
	    delrx=(mj==1.0?delrxA:delrxB);

	    dat deltax=bi.x-bj.x;
            dat deltay=bi.y-bj.y;

	    /* special LE shift */
 	    deltax=(deltay>ly*0.5f?deltax-delrx:deltax);	/* if dy/ly > 0.5, particle j moves to the right (dx -= delrx) */
 	    deltax=(deltay<-ly*0.5f?deltax+delrx:deltax);	/* if dy/ly < 0.5, particle j moves to the left (dx += delrx) */

// 	    deltax=deltax-lx*roundf(deltax*invlx);	/* V14+! very important (for running coordinates?)*/
//          deltay=deltay-ly*roundf(deltay*invly);
	     /* standard Image Convention via inline */
            deltax=(deltax>lx*0.5f?deltax-lx:deltax);
            deltax=(deltax<-lx*0.5f?deltax+lx:deltax);
            deltay=(deltay>ly*0.5f?deltay-ly:deltay);
            deltay=(deltay<-ly*0.5f?deltay+ly:deltay);

	    /* calculate squared distance */
            dat rijsq=deltax*deltax+deltay*deltay;
// 	    rijsq  += (1-overlap)*rcut*rcut;
            if(rijsq<rcut*rcut)
            {
		  /* rsqrt() is the reciprocal square root.
		  * add the cut [-vpcutbd] force term. Multiply by 1/r to get the components right, see below */
                  dat rij=rsqrt(rijsq)*(1.0f/(rijsq*rijsq)-vpcutbd);
		  rij *= mi*mj;
		  /* the magnitude of the acceleration is [1.0/(rijsq*rijsq)-vpcutbd].
		  * Obtain the x,y-components by multiplying by dx/r or dy/r, respectively */
                  ai.x+=rij*deltax;
                  ai.y+=rij*deltay;
            }
  /* ai is initialized before being passed to this function. So in case no interaction was added,
  * the acceleration is still defined */
  return ai;
}
/*************************************************************************************************/


__global__ void setup_random_numbers(curandState *devStates,const unsigned long long seed,const int Np)
{
/************************************************************************
* Description:								*
*************************************************************************
* initializes random numbers for each thread				*
*									*
* Written by CVA							*
*************************************************************************
* Arguments: 								*
*************************************************************************
* *devStates: random states for each thread				*
* seed: random seed							*
* N: particle number							*
************************************************************************/
  int id = threadIdx.x + blockIdx.x * tpb;	/* unique thread id < N */
  if(id<Np)
      {
	/* Each thread gets a different or identical seed, a different sequence number,no offset */
//      curand_init(seed+id*rsrs, id, 0, &devStates[id]);
      curand_init(seed, id, 0, &devStates[id]);
      }
}
/*************************************************************************************************/


__global__ void config_init(curandState *devStates, dat2 *result, const int Np,const dat lx,const dat ly)
{
/************************************************************************
* Description:								*
*************************************************************************
* initializes starting configuration				*
*									*
* Written by CVA							*
*************************************************************************
* Arguments: 								*
*************************************************************************
* *devStates: random states for each thread				*
* *result: 2D starting configuration entries				*
* Np: particle number							*
* lx(ly): box x(y)-dimension in native units				*
************************************************************************/
  int id = threadIdx.x + blockIdx.x * tpb;	/* unique thread id < N */

  const int nrows = (int)sqrt((double)Np);	/* There are nrows rows of each species. */
  const int NA = 0.5*Np;			/* the total number of particles per species */
  const dat rowfac = 1;  			/* with LX = sqrt(NA), the height of a row is 1 */
  const dat lyA = rowfac*(dat)nrows;		/* height of the substrate (derived from nrows) */
  const int NAcut = NA*lyA/ly;			/* number of A-particles in the substrate */
  const int NArow = (int)sqrt(0.5*(double)NA);	/* number of A-particles per row (16) */

  if(id<Np){
      dat2 locpos;
      /* Copy state to local memory for efficiency */
      curandState localState = devStates[id];
      /* Generate pseudo-random unsigned ints */

      /* free particles are distributed in the space above the substrate */
      if (id>=2*NAcut)
      {
	locpos.x = curand_uniform(&localState)*lx - 0.5*lx;
 	locpos.y = curand_uniform(&localState)*(ly-lyA)+lyA - 0.5*ly;
// 	locpos.y = curand_uniform(&localState)*(0.5*ly)+0.5*ly;

      }
      /* substrate particles are arranged in an SqA-SqB-pattern */
      else
      {
	int species = id%2; /* even id --> species A, odd id --> species B */
	int myid;	    /* effective id of a particle within its species */
	myid=(species==0?0.5*id:id-(int)floor((double)0.5*id)-1); /* sure about -1? --> yes, first index is 0! */
	int xbin = myid%NArow;				/* species-dependent particle index within its row */
	int ybin = floor((double)myid/NArow); 		/* species-dependent row index */
	locpos.x = xbin + (1-species)*0.5 - 0.5*lx;		/* square lattice is shifted by 0.5*aA in x&y for species B */
	locpos.y = ybin + (1-species)*0.5 - 0.5*ly;
      }
      /* reduce to box by applying PBCs */
      locpos.x=locpos.x-lx*round((double)locpos.x/(double)lx);
      locpos.y=locpos.y-ly*round((double)locpos.y/(double)ly);
      /* Copy devStates back to global memory */
      devStates[id] = localState;
      /* Store results */
      result[id] = locpos;
      }
    if(id==Np){
      dat2 locpos;
      locpos.x = 0.0;
      locpos.y = 0.0;
      result[id] = locpos;
    }


}

__global__ void config_cut(dat2 *devpos, const int Np,const dat lx,const dat ly, const dat invlx, const dat invly)
{
/************************************************************************
* Description:								*
*************************************************************************
* takes a configuration and cuts all coordinates according to PBCs.	*
* This is used before initializing a new dataset with an existing 	*
* configuration.							*
*									*
* Written by TH								*
*************************************************************************
* Arguments: 								*
*************************************************************************
* *pos: configuration entries						*
* Np: particle number							*
* lx(ly): box x(y)-dimension in native units				*
************************************************************************/
  int id = threadIdx.x + blockIdx.x * tpb;	/* unique thread id < N */

  dat2 locpos;

  locpos = devpos[id];
  locpos.x=locpos.x-lx*round((double)locpos.x*(double)invlx);
  locpos.y=locpos.y-ly*round((double)locpos.y*(double)invly);

  /* Store results */
  devpos[id] = locpos;

}

__global__ void increment_time(dat2 *devpos, const int Np, const dat plustime)
{
  int id = threadIdx.x + blockIdx.x * tpb + 1;	/* unique thread id 1..N */

  dat2 locpos;
  if (id == Np) {
  locpos = devpos[id];
  locpos.x = locpos.x + plustime;
  /* Store results */
  devpos[id] = locpos;
  }

}
/*************************************************************************************************/
__global__ void binary_init(dat2 *devbin, const int Np)
{
/************************************************************************
* Description:								*
*************************************************************************
* initializes matrix of binary properties m,D				*
*									*
* Written by TH								*
*************************************************************************
* Arguments: 								*
*************************************************************************
* *devbin: binary properties m,D of each particles			*
* Np: particle number							*
************************************************************************/
  int id = threadIdx.x + blockIdx.x * tpb;	/* unique thread id < N */
  if(id<Np)
      {
      dat2 locbin;
      /* assign dipole moment and diffusion constant for species A (i=0..N/2-1) and B (i=N/2..N-1) */
      locbin.x = (id%2==1?0.1:1.0);
      locbin.y = (id%2==1?1.7:1.0);
      /* Store results */
      devbin[id] = locbin;
      }
}
/*************************************************************************************************/


__global__ void
__launch_bounds__(tpb,minBlocksPerMultiprocessor)
forces_calculation_tiles(curandState *devStates,dat2 *position, dat2 *dR, dat2 *devbin,const int Np,
                            const dat lx,const dat ly,const dat vpcutbd,const dat rcut,
                            const dat dt,const dat ggamma,const dat sqtwodt, const dat sqrtDB,const int devGO, const dat invlx, const dat invly, const dat DB, const dat shear_rate,
			    const dat delrxA, const dat delrxB)
{
  int ip = threadIdx.x + blockIdx.x * tpb;
  int species;
  int species_id;
  dat gamma_i;
  dat delrx;
  dat sqrtD;
  dat yabs;
  double2 myacc;
  myacc.x=0.0;
  myacc.y=0.0;
  dat2 myposip=position[ip];
  dat2 mybin=devbin[ip];
  dat2 mydR=dR[ip];
  curandState localState = devStates[ip];
  __shared__ dat2 shposition[tpb];
  __shared__ dat2 shneibcellbin[tpb];
//#pragma unroll
  for (int tile = 0; tile < bpg;tile++)
  {
      int jp = tile *tpb + threadIdx.x;
      shposition[threadIdx.x] = position[jp];
      shneibcellbin[threadIdx.x] = devbin[jp];
      __syncthreads();
//#pragma unroll
      for(int iloc=0;iloc<tpb;iloc++)
      {
           jp=iloc+tpb*tile;
//
 	   if(ip!=jp)
                {
            myacc=dipolarInteraction(myposip,shposition[iloc],myacc,rcut,vpcutbd,lx,ly,mybin.x,shneibcellbin[iloc].x,invlx,invly,delrxA,delrxB);
                }
       }
       __syncthreads();
  }

     dat2 rndnormal=curand_normal2(&localState);			/* get new random state */
     species = ip%2;
     sqrtD = (species==0?1.0:sqrtDB);
     gamma_i = 1.0;	/* !!! homogeneous test case !!! */
//      gamma_i = (species==0?DB:1.0);
     delrx = (species==0?delrxA:delrxB);	/* delrxA = delrxB globally, test case */
     species_id=(species==0?0.5*ip:ip-(int)floor((double)0.5*ip)-1);

     yabs = myposip.y + 0.5*ly;

     dat deltax = mybin.y*3.0f*ggamma*dt*myacc.x + sqrtD*sqtwodt*rndnormal.x + gamma_i*shear_rate*yabs*dt;
     dat deltay = mybin.y*3.0f*ggamma*dt*myacc.y + sqrtD*sqtwodt*rndnormal.y;

     myposip.x = myposip.x + deltax;
     myposip.y = myposip.y + deltay;

     mydR.x = mydR.x + deltax;
     mydR.y = mydR.y + deltay;

     /* IMAGE CONVENTION AT PARTICLE UPDATE */

     /* Lees-Edwards part */
     myposip.x=myposip.x-round((double)myposip.y/(double)ly)*delrx; 	/* shift particle in x position if it crosses y-boundary */
     /* standard PBCs */

      myposip.x=myposip.x-lx*round((double)myposip.x/(double)lx);	/* replace particle with image particle for convenience */
//      myposip.x=(myposip.x<-0.5*lx?myposip.x+lx:myposip.x);	/* safety inline */
//      myposip.x=(myposip.x>=0.5*lx?myposip.x-lx:myposip.x);	/* safety inline */
      myposip.y=myposip.y-ly*round((double)myposip.y/(double)ly);
//      myposip.y=(myposip.y<-0.5*ly?myposip.y+ly:myposip.y);	/* safety inline */
//      myposip.y=(myposip.y>=0.5*ly?myposip.y-ly:myposip.y);	/* safety inline */

     position[ip] = myposip;		/* update position on device */
     dR[ip] = mydR;
     devStates[ip] = localState;
}

/*************************************************************************************************/

void create_dataset(dat2 *config, dat2 *dR, const int idx, char *myfilename)
{
   	/*
    	* This routine creates a hdf5-dataset
	* and closes it afterwards. The dataset is saved to FILENAME and contains
	* dataspace for a 3D-array MAXCONF*MAXPART*MAXDIMS and several attributes,
	* such as the parameter list and names.
	*
	* This and the following hdf5-operations are composed of the basic examples
	* provided at http://www.hdfgroup.org/HDF5/Tutor/introductory.html
	*/

   hid_t       file_id, dataset_id, dataset2_id, dataspace_id, dataspace2_id, aid1, aid2, aid4;  /* identifiers */
   hid_t       a1dataspace_id, a2dataspace_id, a4dataspace_id, a2type;
   hsize_t     adim,size;
   herr_t      status;

    hid_t        prop;

    hsize_t      dims[3];           /* dataset dimensions at creation time */
    dims[0] = 1;
    dims[1] = PARTNUMBER+1;
    dims[2] = MAXDIMS;
    hsize_t      maxdims[3] = {H5S_UNLIMITED, PARTNUMBER+1, MAXDIMS};
    hsize_t      chunk_dims[3] = {1, PARTNUMBER+1, MAXDIMS};

 /* Create the data space with unlimited dimensions. */
    dataspace_id = H5Screate_simple (3, dims, maxdims);


    /* Create a new file. If file exists its contents will be overwritten. */
    file_id = H5Fcreate (myfilename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    /* Modify dataset creation properties, i.e. enable chunking  */
    prop = H5Pcreate (H5P_DATASET_CREATE);
    status = H5Pset_chunk (prop, 3, chunk_dims); /* chunk is effectively! 2D */
    /* Create a new dataset within the file using chunk
       creation properties.  */
    dataset_id = H5Dcreate2 (file_id, "/set", H5T_IEEE_F64LE, dataspace_id,
                         H5P_DEFAULT, prop, H5P_DEFAULT);

    /* Write data to dataset */
    status = H5Dwrite (dataset_id, H5FORMAT, H5S_ALL, H5S_ALL,
                       H5P_DEFAULT, config);

    /* Repeat for additional dataset which contains the (total) displacement from starting position */
    /**********************/
    dataspace2_id = H5Screate_simple (3, dims, maxdims);
    dataset2_id = H5Dcreate2 (file_id, "/dR", H5T_IEEE_F64LE, dataspace2_id,
			    H5P_DEFAULT, prop, H5P_DEFAULT);
    status = H5Dwrite (dataset2_id, H5FORMAT, H5S_ALL, H5S_ALL,
			  H5P_DEFAULT, dR);
    /**********************/

   /* Create the data space for attribute 1. */
   adim = MAXPRM;
   a1dataspace_id = H5Screate_simple(1, &adim, NULL);
   /* Create dataset attribute "prm". */
   aid1 = H5Acreate(dataset_id,"prm", H5T_IEEE_F64LE, a1dataspace_id,
                             H5P_DEFAULT, H5P_DEFAULT);

   /* Create the data space for attribute 2. */
   a2dataspace_id = H5Screate_simple(1, &adim, NULL);
   a2type = H5Tcopy(H5T_C_S1);
   size = STRLEN;
   status = H5Tset_size(a2type, size);
   /* Create dataset attribute "prmname". */
   aid2 = H5Acreate(dataset_id,"prmname", a2type, a2dataspace_id,
                             H5P_DEFAULT, H5P_DEFAULT);

   /* Create the data space for attribute 4. */
   a4dataspace_id  = H5Screate(H5S_SCALAR);
   /* Create dataset attribute "index". */
   aid4 = H5Acreate(dataset_id, "index", H5T_NATIVE_INT, a4dataspace_id,
                     H5P_DEFAULT,H5P_DEFAULT);

   /* Write the attribute */
   status = H5Awrite(aid1, H5T_NATIVE_DOUBLE, PRM);
   status = H5Awrite(aid2, a2type, PRMNAME);
   status = H5Awrite(aid4, H5T_NATIVE_INT, &idx);

   /* Terminate access to the attribute data space. */
   status = H5Sclose(a1dataspace_id);
   status = H5Sclose(a2dataspace_id);
   status = H5Sclose(a4dataspace_id);
   /* Terminate access to the data space. */
   status = H5Sclose(dataspace_id);
   status = H5Sclose(dataspace2_id);
   /* Close the attribute. */
   status = H5Aclose(aid1);
   status = H5Aclose(aid2);
   status = H5Aclose(aid4);
   /* End access to the dataset and release resources used by it. */
   status = H5Dclose(dataset_id);
   status = H5Dclose(dataset2_id);
   /* Close the file. */
   status = H5Fclose(file_id);
}
/*************************************************************************************************/

void add_config(dat2 *config, const int idx, char *myfilename)
{
	hid_t        file_id;                          /* handles */
	hid_t	     dataset_id;
	hid_t        aid4;
    	herr_t       status;
    	hid_t        filespace, memspace;
	hsize_t      size[3];
	hsize_t      dimsr[3];
   	hsize_t      offset[3];
    	hsize_t      dimsext[3] = {1, PARTNUMBER+1, MAXDIMS};         /* extend dimensions */
	int rank;


	/* Open an existing file. */
    	file_id = H5Fopen(myfilename, H5F_ACC_RDWR, H5P_DEFAULT);
	 /* Open an existing dataset. */
    	dataset_id = H5Dopen2(file_id, "/set", H5P_DEFAULT);
	/* some new operations related to the chunked, extendible dataset */
	filespace = H5Dget_space (dataset_id);				/* mine */
	rank = H5Sget_simple_extent_ndims (filespace); 			/* mine */
	status = H5Sget_simple_extent_dims (filespace, dimsr, NULL);	/* mine */
	/* Extend the dataset */
	size[0] = dimsr[0]+1;
     	size[1] = dimsr[1];
 	size[2] = dimsr[2];
    	status = H5Dset_extent (dataset_id, size);

    	/* Select a hyperslab in extended portion of dataset  */
    	filespace = H5Dget_space (dataset_id);				/* necessary ? with mine(1) ? */

    	offset[0] = dimsr[0]; 	/* 3 in example h5_extent.cu. My example: Configs 0,1,2 written, want to write idx = 3, so there are already 3 entries (offset) */
    	offset[1] = 0;		/* no offset in particles */
	offset[2] = 0;		/* no offset in dims */

	status = H5Sselect_hyperslab (filespace, H5S_SELECT_SET, offset, NULL,
                                  dimsext, NULL);

   	 /* Define memory space */
  	 memspace = H5Screate_simple (3, dimsext, NULL);

    	/* Write the data to the extended portion of dataset  */
    	status = H5Dwrite (dataset_id, H5FORMAT, memspace, filespace,
                       H5P_DEFAULT, config);

 	/* Open an existing attribute. */
  	aid4 = H5Aopen(dataset_id,"index", H5P_DEFAULT);
 	/* Write the attribute */
  	status = H5Awrite(aid4, H5T_NATIVE_INT, &idx);

    	/* Close the dataset. */
    	status = H5Dclose(dataset_id);
    	/* Close the attribute. */
  	status = H5Aclose(aid4);
    	/* Close the file. */
    	status = H5Fclose(file_id);
}
/*************************************************************************************************/
void add_dR(dat2 *dR, const int idx, char *myfilename)
{
	hid_t        file_id;                          /* handles */
	hid_t	     dataset_id;
    	herr_t       status;
    	hid_t        filespace, memspace;
	hsize_t      size[3];
	hsize_t      dimsr[3];
   	hsize_t      offset[3];
    	hsize_t      dimsext[3] = {1, PARTNUMBER+1, MAXDIMS};         /* extend dimensions */
	int rank;


	/* Open an existing file. */
    	file_id = H5Fopen(myfilename, H5F_ACC_RDWR, H5P_DEFAULT);
	 /* Open an existing dataset. */
    	dataset_id = H5Dopen2(file_id, "/dR", H5P_DEFAULT);
	/* some new operations related to the chunked, extendible dataset */
	filespace = H5Dget_space (dataset_id);				/* mine */
	rank = H5Sget_simple_extent_ndims (filespace); 			/* mine */
	status = H5Sget_simple_extent_dims (filespace, dimsr, NULL);	/* mine */
	/* Extend the dataset */
	size[0] = dimsr[0]+1;
     	size[1] = dimsr[1];
 	size[2] = dimsr[2];
    	status = H5Dset_extent (dataset_id, size);

    	/* Select a hyperslab in extended portion of dataset  */
    	filespace = H5Dget_space (dataset_id);				/* necessary ? with mine(1) ? */

    	offset[0] = dimsr[0]; 	/* 3 in example h5_extent.cu. My example: Configs 0,1,2 written, want to write idx = 3, so there are already 3 entries (offset) */
    	offset[1] = 0;		/* no offset in particles */
	offset[2] = 0;		/* no offset in dims */

	status = H5Sselect_hyperslab (filespace, H5S_SELECT_SET, offset, NULL,
                                  dimsext, NULL);

   	 /* Define memory space */
  	 memspace = H5Screate_simple (3, dimsext, NULL);

    	/* Write the data to the extended portion of dataset  */
    	status = H5Dwrite (dataset_id, H5FORMAT, memspace, filespace,
                       H5P_DEFAULT, dR);

    	/* Close the dataset. */
    	status = H5Dclose(dataset_id);
    	/* Close the file. */
    	status = H5Fclose(file_id);
}
/*************************************************************************************************/

void read_config(dat2 *config, const int Np, const int idx_top, const int idx_read, char *myfilename)
 {


	/****************************************************************/
 	hid_t       file_id, dataset_id;  /* identhandle.ifiers */
 	hid_t       aid1,aid2,a2type;
     	herr_t      status;
 	hsize_t     size;
	hid_t       filespace, memspace;
	hid_t        prop;

 	hsize_t      chunk_dimsr[3], offset[3];
    	hsize_t      dimsr[3];

// 	int 	    i;
        int         rank;
 	int	    rank_chunk;

 	/* Open an existing file. */
    	file_id = H5Fopen(myfilename, H5F_ACC_RDONLY, H5P_DEFAULT);

    	/* Open an existing dataset. */
    	dataset_id = H5Dopen2(file_id, "/set", H5P_DEFAULT);
	/* some new operations related to the chunked, extendible dataset */
	filespace = H5Dget_space (dataset_id); 			/* new: read filespace */
	rank = H5Sget_simple_extent_ndims (filespace); 		/* new: read rank */
	/*
	 * read dimensions of dataset
	 */
	status = H5Sget_simple_extent_dims (filespace, dimsr, NULL);	/* read dims back */
	const int d0 = (int)dimsr[0];	/* convert read dimensions to int for output */
	const int d1 = (int)dimsr[1];
// 	const int d2 = (int)dimsr[2];

	/* error checks */
	if (idx_read > d0 || idx_read < 0) {
		printf("read_config(): Error, illegal read index %d (top index: %d)\n", idx_read,d0);
		exit(1);
	}
	if (Np != d1 - 1 || Np < 0) {
		printf("read_config(): Error, illegal particle number %d (corresponding dataset dim: %d)\n", Np,d1-1);
		exit(1);
	}

	prop = H5Dget_create_plist (dataset_id);
 	if (H5D_CHUNKED == H5Pget_layout (prop)) {
     			rank_chunk = H5Pget_chunk (prop, rank, chunk_dimsr); }

	offset[0] = idx_read;
    	offset[1] = 0;		/* no offset in particles */
	offset[2] = 0;		/* no offset in dims */

	status = H5Sselect_hyperslab (filespace, H5S_SELECT_SET, offset, NULL,
                                  chunk_dimsr, NULL);

   	 /* Define memory space */
  	 memspace = H5Screate_simple (rank, chunk_dimsr, NULL);

	status = H5Dread (dataset_id, H5FORMAT, memspace, filespace, H5P_DEFAULT, config);

    	status = H5Pclose (prop);
    	status = H5Sclose (filespace);
    	status = H5Sclose (memspace);

 	/* Open an existing attribute. */
 	aid1 = H5Aopen(dataset_id,"prm", H5P_DEFAULT);
 	aid2 = H5Aopen(dataset_id,"prmname", H5P_DEFAULT);

 	a2type = H5Tcopy(H5T_C_S1);
    	size = STRLEN;
 	status = H5Tset_size(a2type, size);

 	/* Read the attribute */
 	status = H5Aread(aid1, H5T_NATIVE_DOUBLE, PRM); //buffer is global PRM
 	status = H5Aread(aid2, a2type, PRMNAME);	//buffer is global PRMNAME

    	/* Close the dataset. */
    	status = H5Dclose(dataset_id);
    	/* Close the attribute. */
    	status = H5Aclose(aid1);
 	status = H5Aclose(aid2);

    	/* Close the file. */
    	status = H5Fclose(file_id);

}
/*************************************************************************************************/

void read_dR(dat2 *dR, const int Np, const int idx_top, const int idx_read, char *myfilename)
 {


	/****************************************************************/
 	hid_t       file_id, dataset_id;  /* identhandle.ifiers */
//  	hid_t       aid1,aid2,a2type;
     	herr_t      status;
//  	hsize_t     size;
	hid_t       filespace, memspace;
	hid_t        prop;

 	hsize_t      chunk_dimsr[3], offset[3];
    	hsize_t      dimsr[3];

// 	int 	    i;
        int         rank;
 	int	    rank_chunk;

 	/* Open an existing file. */
    	file_id = H5Fopen(myfilename, H5F_ACC_RDONLY, H5P_DEFAULT);

    	/* Open an existing dataset. */
    	dataset_id = H5Dopen2(file_id, "/dR", H5P_DEFAULT);
	/* some new operations related to the chunked, extendible dataset */
	filespace = H5Dget_space (dataset_id); 			/* new: read filespace */
	rank = H5Sget_simple_extent_ndims (filespace); 		/* new: read rank */
	/*
	 * read dimensions of dataset
	 */
	status = H5Sget_simple_extent_dims (filespace, dimsr, NULL);	/* read dims back */
	const int d0 = (int)dimsr[0];	/* convert read dimensions to int for output */
	const int d1 = (int)dimsr[1];
// 	const int d2 = (int)dimsr[2];

	/* error checks */
	if (idx_read > d0 || idx_read < 0) {
		printf("read_config(): Error, illegal read index %d (top index: %d)\n", idx_read,d0);
		exit(1);
	}
	if (Np != d1 - 1 || Np < 0) {
		printf("read_config(): Error, illegal particle number %d (corresponding dataset dim: %d)\n", Np,d1-1);
		exit(1);
	}

	prop = H5Dget_create_plist (dataset_id);
 	if (H5D_CHUNKED == H5Pget_layout (prop)) {
     			rank_chunk = H5Pget_chunk (prop, rank, chunk_dimsr); }

	offset[0] = idx_read;
    	offset[1] = 0;		/* no offset in particles */
	offset[2] = 0;		/* no offset in dims */

	status = H5Sselect_hyperslab (filespace, H5S_SELECT_SET, offset, NULL,
                                  chunk_dimsr, NULL);

   	 /* Define memory space */
  	 memspace = H5Screate_simple (rank, chunk_dimsr, NULL);

	status = H5Dread (dataset_id, H5FORMAT, memspace, filespace, H5P_DEFAULT, dR);

    	status = H5Pclose (prop);
    	status = H5Sclose (filespace);
    	status = H5Sclose (memspace);

    	/* Close the dataset. */
    	status = H5Dclose(dataset_id);

    	/* Close the file. */
    	status = H5Fclose(file_id);

}
/*************************************************************************************************/
void output(dat2 *config, const int Np, const double L, const int idx)
{
	int j;
	FILE *fpO;
	double xj,yj;
	char outfile[50];
	sprintf(outfile,"out%d.dat", idx);
	if ((fpO = fopen(outfile, "w")) == NULL) {
		fprintf(stderr,"Could not open file '%s'.\n", outfile);
		exit(1);
	}
	for (j = 0; j < Np; j++) {
		xj = config[j].x; xj -= floor((double)xj / L) * L;	/* shift to box */
		yj = config[j].y; yj -= floor((double)yj / L) * L;	/* shift to box */
		fprintf(fpO,"%lf\t%lf\n",xj,yj);
	}
	fclose(fpO);

}
/*************************************************************************************************/


void read_idx(int * idx, char *myfilename) {
 	hid_t       file_id, dataset_id;  /* identhandle.ifiers */
 	hid_t       aid4;
    	herr_t      status;
	int 	    idx_temp;

 	/* Open an existing file. */
    	file_id = H5Fopen(myfilename, H5F_ACC_RDONLY, H5P_DEFAULT);

    	/* Open an existing dataset. */
    	dataset_id = H5Dopen2(file_id, "/set", H5P_DEFAULT);

 	/* Open an existing attribute. */
 	aid4 = H5Aopen(dataset_id,"index", H5P_DEFAULT);

 	/* Read the attribute */
 	status = H5Aread(aid4, H5T_NATIVE_INT, &idx_temp);	//buffer is idx_temp

    	/* Close the dataset. */
    	status = H5Dclose(dataset_id);
    	/* Close the attribute. */
 	status = H5Aclose(aid4);

    	/* Close the file. */
    	status = H5Fclose(file_id);
  	*idx = idx_temp;
}
void assign_prm(double value, char key[STRLEN])
{
	int i;
	int ret = 0;
	for (i = 0; i < MAXPRM; i++){
		if (strcmp(key,PRMNAME[i]) == 0) {
			PRM[i] = value;
			ret ++ ;
			}
		}
	if (ret != 1) {
	fprintf(stderr,"PRM ASSIGN ERROR: Key %s could not be matched correctly.\n", key);
	exit(1);
	}
}
/*************************************************************************************************/

double get_prm(char key[STRLEN])
{
	int i;
	int ret = 0;
	double value; //return value
	for (i = 0; i < MAXPRM; i++){
		if (strcmp(key,PRMNAME[i]) == 0) {
			value = PRM[i];
			ret ++ ;
			}
		}
	if (ret != 1) {
	fprintf(stderr,"PRM GET ERROR: Key %s could not be matched correctly.\n", key);
	exit(1);
	}
	return value;
}
/*************************************************************************************************/


void init_prm()
{
	sprintf(PRMNAME[0],"dummy");	PRM[0] = VERSION;	/* the name of the simulation run / the version number */
	sprintf(PRMNAME[1],"none");	PRM[1] = 0.0;		/* the preceeding run and the config no. from which this was taken */
	sprintf(PRMNAME[2],"Gamma");	PRM[2] = 53.0;		/* the dimensionless coupling constant Gamma */
	sprintf(PRMNAME[3],"N");	PRM[3] = 131072.0;	/* the number of particles (rewritten by main()) */
	sprintf(PRMNAME[4],"tpb");	PRM[4] = tpb;		/* the number threads per blocks */
	sprintf(PRMNAME[5],"bpg");	PRM[5] = bpg;		/* the number of blocks */
	sprintf(PRMNAME[6],"L");	PRM[6] = 0.0;		/* the box dimension (calculated) */
	sprintf(PRMNAME[7],"INTVL");	PRM[7] = 1.0;		/* the number of iterations until a configuration is saved */
	sprintf(PRMNAME[8],"RT");	PRM[8] = 10.0;		/* the total number of iterations */
	sprintf(PRMNAME[9],"D");   	PRM[9] = 1.7;		/* ratio of diffusion constants DB/DA --> based on*/
	sprintf(PRMNAME[10],"m");   	PRM[10] = 0.1;		/* ratio of dipole moments DB/DA */
	sprintf(PRMNAME[11],"F");   	PRM[11] = 0.001;		/* shear rate */

	/* Version message */
	fprintf(stderr, "\ncudabd version %.3lf.\n", PRM[0]);
}
/*************************************************************************************************/


void get_args(int argc, char** argv, int *task, int *info)
{
    int i;

    /* Start at i = 1 to skip the command name. */

    for (i = 1; i < argc; i++) {

	/* Check for a switch (leading "-"). */

	if (argv[i][0] == '-') {

	    /* Use the next character to decide what to do. */

	    switch (argv[i][1]) {

		/* Use these switches to overwrite parameters */
		case 'G':	assign_prm(atof(argv[++i]),"Gamma");
				break;
		case 'I':	assign_prm(atof(argv[++i]),"INTVL");
				break;
		case 'R':	assign_prm(atof(argv[++i]),"RT");
				break;
		case 'D':	assign_prm(atof(argv[++i]),"D");
 				break;
		case 'm':	assign_prm(atof(argv[++i]),"m");
 				break;
		case 'F':	assign_prm(atof(argv[++i]),"F");
 				break;
		case 'S':	strcpy(PRMNAME[0],argv[++i]);
				break;

		/* Use these switches to determine the task */
		case 'i':	*task = 0; /* initialise and write new file */
				break;
		case 'r':	*task = 1; /* read existing file and run from latest config */
				break;
 		case 'x':	*task = 2; /* produce test output */
//  				if (argv[++i] != NULL) { *info = atoi(argv[i]); }
 				break;
		case 'c':	*task = 3; /* copy from FILENAME (last index) to FILENAME2 (index 0)*/
				if (argv[++i] != NULL) { strcpy(PRMNAME[0],argv[i]); }
				if (argv[++i] != NULL) {
					for (int k=0; k<nstreams; k++) {
					char ktag[2];
					/* fit stream tag to two digits */
					if (k >= 9) { sprintf(ktag,"%d",k+1);}
					else {sprintf(ktag,"0%d",k+1);}

					sprintf(FILENAME2[k],".",argv[i],ktag);
// 					fprintf(stderr,"Filename2[%d] %s\n",k,FILENAME2[k]);
					}
// 					sprintf(FILENAME2_STREAM2,"/local/tobias/sim/004_cumyle/%s-s2.h5",argv[i]);
				}

				break;
		case 'f':	*task = 4; /* perform one sequential iteration (no random stuff) and compare to gpu results */
				break;
		case 'v':	*task = 5; /* perform one sequential iteration (no random stuff) and compare to gpu results */
				break;
		case 'g':	GO = 1; /* set global variable "GO" to one and release lower half of system */
				break;

		default:	fprintf(stderr,
				"Unknown switch: %s\n", argv[i]);
	    }
	}
    }


	if ((int)get_prm("RT") % (int)get_prm("INTVL") != 0) {
		fprintf(stderr,"Uneven ratio of RT & INTVL! Exit.\n");
 		exit(1);
	}
}
/*************************************************************************************************/


void read_check(dat2 *host, dat2 *read, const int Np) {

	int i;
	int nerr = 0;
	dat2 diff;
	dat d;
	diff.x = 0.0;
	diff.y = 0.0;
	for (i=0; i<Np; i++){
		fprintf(stderr,"Particle %d:\t%lf\t%lf\n",i,host[i].x,host[i].y);
		printf("\n");
	}
}
/*************************************************************************************************/


int main(int argc, char *argv[])
{

/* old main with task-reading and hdf ability*/
//     cudaSetDevice(0);		/* 0 for Michelangelo */
//    int idx=0;			/* initialise index variable */
   int idx[nstreams];
   int task = 55;		/* initialise default task (default: unknown switch)*/
   int info = 0;		/* initialise test output index */
//    const int safety_read = 0;	/* in safety mode, every written configuration is immemediately re-read and compared with the original one --> doubles buffer size */
   GO = 0;
   int devGO;
   int i;

   for (i=0;i<nstreams;i++) { idx[i] = 0;}


   /*****************************************************
   * initialise parameter list	       			*
   ******************************************************/
   init_prm();								/* assign fail-safe values to parameters */
   get_args(argc, argv, &task, &info); 					/* Get parameter ucuda allocate space on device check capacitypdates from command line */
   for (i=0;i<nstreams;i++){
   char itag[2];
   if (i >= 9) { sprintf(itag,"%d",i+1);}
   else {sprintf(itag,"0%d",i+1);}
   sprintf(FILENAME[i],"%s-s%s.h5",PRMNAME[0],itag);   /* write proper filename  */
//    fprintf(stderr,"\tFilename[%d] %s\n",i,FILENAME[i]);
   }

//    sprintf(FILENAME_STREAM2,"/local/tobias/sim/004_cumyle/%s-s2.h5",PRMNAME[0]);   /* write proper filename  */
    printf("Starting cudash_nbody with task %d\n",task);
    devGO = GO;

   /*****************************************************
   * derive system properties from parameters		*
   ******************************************************/
   const int Np = tpb*bpg;			/* total particle number: (total blocks) times (threads per block) */
   assign_prm((dat)PARTNUMBER,"N");			/* write particle number back to parameter list for future reference in data evaluation */

   const int csize = (PARTNUMBER+1)*sizeof(dat2); 	/* size of a configuration of Np particles (two coordinates each) + 1 extra coo pair for time */
   int nout = (int)get_prm("INTVL");	/* iterations per cycle (before output is generated) */
   int nsteps = (int)get_prm("RT")/nout;	/* number of cycles */

   /* derive cell dimensions from Np */
   const dat lx=(dat)sqrt(0.25*PARTNUMBER); 	/* box x length such that a = 1/sqrt(NA/A) = 1 */
   const dat ly=2.0*lx;
   const dat invlx=1.0f/lx;		/* inverse box length */
   const dat invly=1.0f/ly;


   assign_prm(lx,"L");			/* write box dimension back to parameter list */


   /*****************************************************
   * binary properties					*
   ******************************************************/
   const dat DB = get_prm("D");			/* the diffusion constant of the smaller species */
   const dat sqrtDB = sqrt(DB);			/* correction to BD routine for smaller species */

   /*****************************************************
   * cut- & truncated potential presets			*
   ******************************************************/
   dat Ggamma = (dat)get_prm("Gamma");
   const double rcut=(double)(0.5*lx); 		/* cutoff > LX/2 makes no sense and induces anisotropic interaction */
   const double vpcutene=-3.0/pow(rcut,4);	/* cutoff potential gradient for energy calculation*/
   const double ecut=1.0/pow(rcut,3);		/* cutoff potential for energy calculation */
   const double vpcutbd=1.0/pow(rcut,4);	/* cutoff force (without factor 3) */
   /*****************************************************
   * BD-specific settings				*
   ******************************************************/
   const dat dt=0.00001;			/* timestep  */
   const dat sqtwodt=sqrt(2.0*dt);		/* noise term (implicitly: temperature)*/
   /*****************************************************
   * shear-specific settings				*
   ******************************************************/

   dat timestep;				/* the current time in units of tau needed for the L.E.-PBCs */
   dat timestep_aux;				/* auxiliary timestep used for ratchet shift */
   dat shear_rate = (dat)get_prm("F");	/* the shear rate */
   dat delrxA,delrxB;			/* specific box shift */

   /*****************************************************
   * set up  streams					*
   ******************************************************/
   cudaStream_t stream[nstreams];
   fprintf(stderr,"Setting up streams\t");
   for (i = 0; i < nstreams; i++){

      cudaStreamCreate(&stream[i]);

   }
   fprintf(stderr,".. DONE\n");
   /*********************************************************************
   * initialise seed via wall-clock time + mersenne twister		*
   *********************************************************************/
   unsigned long long seed[nstreams];
   unsigned long long aux = 0;
   uint32_t rnd;
   time_t seconds;
   time(&seconds);
   aux = (unsigned long long) seconds; 	/* read wall clock time in seconds */
   init_genrand(aux);			/* init Mersenne Twister with wall clock time */

   for (i = 0; i < nstreams; i++) {

    rnd = genrand_int32();	/* pick a random number for each stream */
    seed[i]= aux + rnd;		/* add random number as seed offset */
//     fprintf(stderr,"rnd %d --> seed %u\n", rnd, seed[i]);
   }



   /*****************************************************
   * CUDA pointers					*
   ******************************************************/
//    /* For stream 1 */
   curandState *devStates[nstreams]; 	/* device copy of cuda (pseudo-)random states */
   dat2 *devpos[nstreams]; 			/* device copy ofreading config %d from file position matrix, 2-component float */
   dat2 *hostpos[nstreams]; 			/* host copy of position matrix */
   dat2 *devdR[nstreams];			/* total displacement dataset */
   dat2 *hostdR[nstreams];			/* host copy of the above */
   dat2 *devbin[nstreams];			/* binary properties (m,D) for each particles */

   /*****************************************************
   * tile implementation				*
   ******************************************************/
   dim3 blocks,threads;
   blocks.x=PARTNUMBER;		/* particle number is stored in blocks.x */
   blocks.y=bpg;		/* number of blocks is stored in blocks.y */
   blocks.z=1;			/* dummy */
   threads.x=tpb; 		/* threads per block is stored in threads.x */
   threads.y=1;			/* dummy */
   threads.z=1;			/* dummy */
   /*****************************************************
   * Allocate space for results & rand states on device	*
   ******************************************************/
   for (i=0; i<nstreams;i++) {
	CUDA_CALL(cudaMalloc((void **)&devpos[i], csize)); 						/* allocate space for devpos on device (NP*sizeof(float2)) */
	CUDA_CALL(cudaMalloc((void **)&devdR[i], csize));
	CUDA_CALL(cudaMalloc((void **)&devbin[i], csize));
	CUDA_CALL(cudaMalloc((void **)&devStates[i], (Np) * sizeof(curandState)));
   }
   /*****************************************************
   * Stream usage: Pinned memory required on host	*
   ******************************************************/
   for (i=0; i<nstreams;i++) {
	CUDA_CALL(cudaMallocHost((void **)&hostpos[i], csize));
	CUDA_CALL(cudaMallocHost((void **)&hostdR[i], csize));
	if (hostpos[i] == 0 || hostdR[i] == 0){
		      printf("ERROR: Out of memory\n");
		      return 1;
	}
   }
   /*****************************************************
   * init pseud-random generator states			*
   ******************************************************/
   for (i=0; i<nstreams;i++) {
      setup_random_numbers<<<bpg, tpb,0,stream[i]>>>(devStates[i],seed[i], (Np));
      binary_init<<<bpg, tpb,0,stream[i]>>>(devbin[i],Np);
   }
   /*****************************************************
   * Check for task and act accordingly			*
   ******************************************************/
   if (task == 55) {printf("** Unknown switch %d\nExit.",task); exit(0);}
   if (task == 0) {
	/* Task 0: Initialize a new (random) starting configuration (on GPU). Transfer this config to Host, safe this config as
	* first hyperslab (first chunk) in the dataset specified by -S flag. Note that the initial configuration is already on
	* the device since it was generated there */
	/*****************************************************
	* set entries to zero and initialize config 	     *
	******************************************************/
	for (i=0; i<nstreams;i++) {
	    CUDA_CALL(cudaMemsetAsync(devpos[i], 0.0, csize));
	    CUDA_CALL(cudaMemsetAsync(devdR[i], 0.0, csize));
	    config_init<<<bpg, tpb,0,stream[i]>>>(devStates[i], devpos[i], Np,lx,ly); /* generate initial config on GPU */
	}
	/*****************************************************
	* copy positions from device to host 	 	     *
	******************************************************/
	for (i=0; i<nstreams;i++) {
	    cudaStreamSynchronize(stream[i]);
	    CUDA_CALL(cudaMemcpyAsync(hostpos[i], devpos[i], csize, cudaMemcpyDeviceToHost,stream[i]));
	    CUDA_CALL(cudaMemcpyAsync(hostdR[i], devdR[i], csize, cudaMemcpyDeviceToHost,stream[i]));
	}
	/*****************************************************
	* get timestep			 	 	     *
	******************************************************/
	timestep = hostpos[0][Np].x;
	printf("===> Starting time %lf\n",timestep);
	printf("===> Check: stream 1 timestep %lf <--> stream 2 timestep %lf\n",hostpos[0][Np].x,hostpos[1][Np].x);
 	display_prm(PRM,PRMNAME,"PARAMETERS before writing");
	printf("\n\n");
	for (i=0; i<nstreams;i++) {
	    create_dataset(hostpos[i],hostdR[i],0,FILENAME[i]); /* create HDF5 dataset (stack of configuration slabs) and write hostpos as config 0 for STREAM 1 */
	}
   }
   if (task == 1) {
	/* Task 1: Read the latest config of the dataset specified by -S flag. and copy it to the device */
	for (i=0; i<nstreams;i++){
	read_idx(&idx[i],FILENAME[i]);		/* read latest index (stream1) */
	}
	/*****************************************************
	* check read idx consistency	 	 	     *
	******************************************************/
	if (idx[0] != idx[1]) {
	  fprintf(stderr,"Read Idx error (Streams): idx stream1: %d, idx stream2: %d\nExit.\n",idx[0],idx[1]);
	}
	/*****************************************************
	* read latest config to hostpos	 	 	     *
	******************************************************/
	for (i=0; i<nstreams;i++){
	    if (i==0) { printf("** reading config %d from file %s\t",idx[i], FILENAME[i]);} 	/* FYI */
	    read_config(hostpos[i],PARTNUMBER,idx[i],idx[i],FILENAME[i]);		/* copy latest config to hostpos (stream 1) */
	    read_dR(hostdR[i],PARTNUMBER,idx[i],idx[i],FILENAME[i]);		/* copy latest state of dR to host (stream 1) */
	     if (i==0) { printf(".. DONE\n"); }
	}
	/*****************************************************
	* copy hostpos to device	 	 	     *
	******************************************************/
	for (i=0; i<nstreams;i++){
	    if (i==0) {printf("*** Copying hostpos[%d] to device\t",i);}
	    cudaStreamSynchronize(stream[i]);	/* probably not necessary since read_config loop is only on CPU */
	    CUDA_CALL(cudaMemcpyAsync(devpos[i], hostpos[i], csize, cudaMemcpyHostToDevice, stream[i])); /* copy hostpos to device */
	    CUDA_CALL(cudaMemcpyAsync(devdR[i], hostdR[i], csize, cudaMemcpyHostToDevice, stream[i])); /* copy hostpos to device */
	    if (i==0) { printf(".. DONE\n"); }
	}
	timestep = hostpos[0][Np].x;
	printf("\n===> pickup time %lf\n\n",timestep);
	printf("===> Check: stream 1 timestep %lf <--> stream 2 timestep %lf\n",hostpos[0][Np].x,hostpos[1][Np].x);
   }
   if (task == 3) {
	/* Task 3: Read the latest config of the dataset FILENAME and copy it to the new dataset FILENAME2, reset time, exit afterwards*/
	/* do for both streams */

	for (i=0; i<nstreams;i++){
	    if (i==0) {printf("Copy from\t%s\nto\t\t%s\n",FILENAME[i], FILENAME2[i]);}
	    read_idx(&idx[i],FILENAME[i]);	/* read latest index */
	}
	double Gamma_new = get_prm("Gamma");				/* read new gamma value from command line */
	double shear_new = get_prm("F");				/* read new gamma value from command line */

	for (i=0; i<nstreams;i++){
	    if (i==0) {fprintf(stderr,"** reading config %d from file %s\t",idx[i], FILENAME[i]);} 	/* FYI */
	    read_config(hostpos[i],PARTNUMBER,idx[i],idx[i],FILENAME[i]);		/* copy latest config to hostpos (and old parameters) */
	    hostpos[i][Np].x = 0;							/* reset time */
 	    if (i==0) { fprintf(stderr,".. DONE *stream %d*\n",i+1); }
	}

	double Gamma_old = get_prm("Gamma");
	double shear_old = get_prm("F");
	/* remains dual */
	char FILE1_TAG[20];
	char FILE2_TAG[20];
	strcpy(FILE1_TAG,PRMNAME[0]);

	printf("\n===> resetting time to %lf\n",hostpos[0][Np].x);
	assign_prm(Gamma_new,"Gamma");
	assign_prm(shear_new,"F");
	/* Get new filetag from path in order to save it as a parameter */
	/* altough there are two files, they have a common tag */
	sscanf(FILENAME2[0], "004_cumyle/%[^-]s", &FILE2_TAG); /* read until "-" */

	strcpy(PRMNAME[0],FILE2_TAG);
	strcpy(PRMNAME[1],FILE1_TAG);

	PRM[1] = idx[0];
	printf("===> changing file tag from %s to %s\n",FILE1_TAG,FILE2_TAG);
	printf("===> changing Gamma from %lf to %lf\n",Gamma_old,Gamma_new);
	printf("===> changing shear rate from %lf to %lf\n",shear_old,shear_new);
	for (i=0; i<nstreams;i++){
	    strcpy(FILENAME[i],FILENAME2[i]);
	}
	/*
	 * copy hostpos to device, cut the coordinates, copy back to host and THEN save in new dataset
	 */

	for (i=0; i<nstreams;i++){
	    CUDA_CALL(cudaMemsetAsync(devpos[i], 0, csize));					/* init devpos */

	    CUDA_CALL(cudaMemcpyAsync(devpos[i], hostpos[i], csize, cudaMemcpyHostToDevice,stream[i])); 		/* copy hostpos to device */

	    config_cut<<<bpg, tpb, 0, stream[i]>>>(devpos[i], Np,lx,ly,invlx,invly); 	      		/* cut coordinates */

	    cudaStreamSynchronize(stream[i]);
	    CUDA_CALL(cudaMemcpyAsync(hostpos[i], devpos[i], csize, cudaMemcpyDeviceToHost,stream[i])); 		/* copy devpos to host */

	    CUDA_CALL(cudaMemsetAsync(devdR[i], 0, csize));						/* set dR to zero for a fresh start */

	    CUDA_CALL(cudaMemcpyAsync(hostdR[i], devdR[i], csize, cudaMemcpyDeviceToHost,stream[i])); 		/* copy devdR to host */

	    create_dataset(hostpos[i],hostdR[i],0,FILENAME[i]); /* create HDF5 dataset (stack of configuration slabs) and write hostpos as config 0 */

// 	    printf(".. DONE *stream %d* \n",i+1);
	}
	/*
	 * done
	 */
	exit(0);
   }
   /* update parameters which might have changed during reading */
   Ggamma = (dat)get_prm("Gamma");
   shear_rate = (dat)get_prm("F");
   const dat dv = (DB-1.0)*shear_rate*ly;	/* difference in box shift velocity for A- & B-particles */
   const dat tc = 0.5/dv;

   /* FYI output */
   display_prm(PRM,PRMNAME,"RUN PARAMETERS");

   printf("\nrcut: %26.20lf\nvpcutene: %26.20lf\necut: %26.20lf\n",rcut,vpcutene,ecut);

   printf("\ntc: %26.20lf (dv = %26.20lf)\n",tc,dv);


   /*
    * memory check
    */
    size_t free, total;
    printf("\n");
    cudaMemGetInfo(&free,&total);	/* retrieve available memory information (bytes) */
    printf("%lu KB free of total %lu KB at the beginning\n",free/1024,total/1024);

    printf("\n[%d]\t t = %lf\t F = %lf\n",idx[0], hostpos[0][Np].x,shear_rate);

    float gputime;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);


    for(int iout=0;iout<nsteps;iout++)
    {
      timestep = hostpos[0][Np].x;		/* get time from latest write to hostpos */
      timestep_aux = timestep + 0.5*tc;
      for(int istep=0;istep<nout;istep++)
      {
	    /* Lees-Edwards box shift */
	    /* my LE */
	    delrxB=shear_rate*ly*timestep;					/* shift between moving boxes for B-particles, gamma_B = 1*/
	    if (shear_rate > 0) {
	    delrxA=delrxB - 0.5*dv*tc + dv*(timestep_aux - tc*floor((double)timestep_aux/(double)tc));	/* ratchet-like oscillating box shift for A-particles */
	    }
	    else {delrxA = delrxB;}

	    delrxA = delrxB;	/* homogeneous test case */

	    delrxA=delrxA - lx*round((double)delrxA/(double)lx);	/* reduce to interval +/-lx (A&T,p.246) */
	    delrxB=delrxB - lx*round((double)delrxB/(double)lx);
	    for (i=0; i<nstreams;i++){
		forces_calculation_tiles<<<bpg,tpb,0,stream[i]>>>(devStates[i],devpos[i],devdR[i],devbin[i],Np,lx,ly,vpcutbd,rcut,dt,Ggamma,sqtwodt,sqrtDB,devGO,invlx,invly,DB,shear_rate,delrxA,delrxB);
	    }
	    timestep += dt;		/* update timestep */
      }


       /* add time to coordinate pos[Np].x */
       for (i=0; i<nstreams;i++){
 	  increment_time<<<bpg, tpb, 0, stream[i]>>>(devpos[i], Np,(dat)nout*dt); 	      		/* add time to coordinate pos[Np].x */
 	  increment_time<<<bpg, tpb, 0, stream[i]>>>(devdR[i], Np,(dat)nout*dt);
       }

//        cudaDeviceSynchronize();		/* without this barrier, results may be copied to host before the propagation step is complete */

      for (i=0; i<nstreams;i++){
 	  cudaStreamSynchronize(stream[i]);	/* solves the problem of premature writing, too. */
	  CUDA_CALL(cudaMemcpyAsync(hostpos[i], devpos[i], csize, cudaMemcpyDeviceToHost,stream[i]));
	  CUDA_CALL(cudaMemcpyAsync(hostdR[i], devdR[i], csize, cudaMemcpyDeviceToHost,stream[i]));
	  idx[i] += 1;
// 	  dat2 temp;
// 	  temp.x = timestep;
// 	  temp.y = 0;
// 	  hostpos[i][Np] = temp;
// 	  hostdR[i][Np] = temp;
      }



      printf("%d\t t = %lf\t F = %lf\n",idx[0],hostpos[0][Np].x,shear_rate);

      /* now that the timestep is updated on device, there is no need to copy hostpos back here, it is just dropped to file */


      for (i=0; i<nstreams;i++){
//  	  if (i == 0) { read_check(hostpos[i],hostpos[i],PARTNUMBER); }
	  add_config(hostpos[i],idx[i],FILENAME[i]);
	  add_dR(hostdR[i],idx[i],FILENAME[i]);
      }

    }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gputime,start,stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop) ;
    printf(" \n");
    printf("Np= %d \n",PARTNUMBER);
    printf("Time = %g \n",  gputime/1000.0f);

    printf(" \n");
/* Cleanup */
  for (i=0; i<nstreams;i++){
    CUDA_CALL(cudaFree(devStates[i]));
    CUDA_CALL(cudaFree(devpos[i]));
    CUDA_CALL(cudaFree(devdR[i]));
    cudaStreamDestroy(stream[i]);
  }

}
