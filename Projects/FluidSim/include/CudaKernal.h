#ifndef CudaKernal_H
#define CudaKernal_H

#include "Particle.h"
#include "FloatPair.h"
#include "IntPair.h"
#include "SpacialIndex.h"


void cuda_kernel();

void __updateParticle(Particle *particles, float *dt, int *NUM_PARTICLES, float *densities, int *h_spatialLookup, SpacialIndex *h_spacialIndexs, 
    FloatPair *h_pressureForce, IntPair *h_offsets, float *h_smoothingRadius, float *h_mass, float *h_pressureMult, float *h_targetDensity);

#endif
