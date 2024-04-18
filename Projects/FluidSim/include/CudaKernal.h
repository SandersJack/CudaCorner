#ifndef CudaKernal_H
#define CudaKernal_H

#include "Particle.h"

void cuda_kernel();

void __updateParticle(Particle *particles, float *dt, int *NUM_PARTICLES, float *densities, int *h_spatialLookup, int *h_startIndex);

#endif
