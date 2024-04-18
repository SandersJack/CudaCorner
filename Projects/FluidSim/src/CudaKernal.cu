#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {
    #include "CudaKernal.h"
}

typedef struct {
    float first;
    float second;
} FloatPair;

int compare(const void *a, const void *b) {
    const float *x = (const float *)a;
    const float *y = (const float *)b;
    if (*x < *y) return -1;
    else if (*x > *y) return 1;
    else return 0;
}

__global__ void kernel() {
    printf("Hello from CUDA kernel!\n");
}

void cuda_kernel() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}

static __device__ float smoothingKernal_new(float radius, float dis){
    if( dis >= radius) return 0;
    
    float volume = (M_PI * pow(radius, 4)) / 6;
    return (radius - dis) * (radius - dis) / volume;
}

static __device__ float smoothingKernalDerivative_new(float radius, float dis){
    if(dis >= radius) return 0;

    float scale = 12 / (pow(radius,4) * M_PI);
    return (dis - radius) * scale;
}

static __device__ void applyGravity(Particle *particles, int id){
    particles[id].dy -= 1;
}

static __device__ void resolveCollisions(Particle *particles, int id){

    int L = 800 - 50;
    float DECAY = 0.5;

    if(particles[id].x > L){
        particles[id].x = L - 0.1;
        particles[id].dx *= -DECAY;
    } else if(particles[id].x < 50) {
        particles[id].x = 50 + 0.1;
        particles[id].dx *= -DECAY;
    }
    if(particles[id].y > L){
        particles[id].y = L - 0.1;
        particles[id].dy *= -DECAY;
    } else if(particles[id].y < 50){
        particles[id].y = 50 + 0.1;
        particles[id].dy *= -DECAY;
    }
}

__global__ void cuda_updateParticle(Particle *particles, float *dt, int *NUM_PARTICLES){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < *NUM_PARTICLES){

        particles[idx].pred_x = particles[idx].x + particles[idx].dx * *dt;
        particles[idx].pred_y = particles[idx].y - particles[idx].dy * *dt;

        applyGravity(particles, idx);
        resolveCollisions(particles, idx);

        particles[idx].x += particles[idx].dx * *dt;
        particles[idx].y -= particles[idx].dy * *dt;

    }
}

void __global__ calculateDensities(float* densities, Particle *particles, int *num_particles){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int MASS = 1;

    if(idx < *num_particles &&  idy < *num_particles){

        float dist_x = particles[idx].pred_x - particles[idy].pred_x;
        float dist_y = particles[idx].pred_y - particles[idy].pred_y;

        float dist = sqrt(dist_x * dist_x + dist_y * dist_y);
        float influence = smoothingKernal_new(100, dist);
        densities[idx] += MASS * influence;

    }
}

static __device__ FloatPair posToCellCoord(Particle particle, float radius){
    FloatPair pair;
    pair.first = (int)(particle.pred_x / radius);
    pair.second = (int)(particle.pred_y / radius);
    return pair;
}

static __device__ int hashCell(int cellX, int cellY){
    int a = cellX * 15823;
    int b = cellY * 9737333;
    return a + b;
}

static __device__ int getKeyFromHash(int *spatialLookup, int hash){
    int sizeOfSL = sizeof(spatialLookup) / sizeof(spatialLookup[0]);
    return hash % sizeOfSL;
}

__global__ void updateSpacialLookup_step1(Particle *particles, int *spatialLookup, int *startIndex, int *NUM_PARTICLES){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    FloatPair cell = posToCellCoord(particles[idx], 100);
    int cellKey = getKeyFromHash(spatialLookup, hashCell(cell.first, cell.second));
    spatialLookup[idx] = cellKey;

    startIndex[idx] = -99;
}

__global__ void updateSpacialLookup_step2(Particle *particles, int *spatialLookup, int *startIndex, int *NUM_PARTICLES){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int key = spatialLookup[idx];
    int keyPrev = (idx == 0) ? -99 : spatialLookup[idx-1];

    if(key != keyPrev){
        startIndex[key] = idx;
    }
}

void __updateParticle(Particle *h_particles, float *h_dt, int *h_NUM_PARTICLES, float *h_densities, int *h_spatialLookup, int *h_startIndex){
    // Allocate memory for the array on the GPU
    Particle *d_particles;
    int *d_NUM_PARTICLES, *d_spatialLookup, *d_startIndex;
    float *d_dt, *d_densities;
    cudaMalloc((void**)&d_particles, *h_NUM_PARTICLES * sizeof(Particle));
    cudaMalloc((void**)&d_NUM_PARTICLES, sizeof(int));
    cudaMalloc((void**)&d_spatialLookup, sizeof(int));
    cudaMalloc((void**)&d_startIndex, sizeof(int));
    cudaMalloc((void**)&d_dt, sizeof(float));
    cudaMalloc((void**)&d_densities, *h_NUM_PARTICLES * sizeof(float));

    cudaMemcpy(d_particles, h_particles, *h_NUM_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_NUM_PARTICLES, h_NUM_PARTICLES, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spatialLookup, h_spatialLookup, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_startIndex, h_startIndex, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dt, h_dt, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_densities, h_densities, *h_NUM_PARTICLES * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (*h_NUM_PARTICLES + blockSize - 1) / blockSize;
    cuda_updateParticle<<<numBlocks, blockSize>>>(d_particles, d_dt, d_NUM_PARTICLES);

    updateSpacialLookup_step1<<<numBlocks, blockSize>>>(d_particles, d_spatialLookup, d_startIndex, d_NUM_PARTICLES);

    cudaMemcpy(h_spatialLookup, d_spatialLookup, *h_NUM_PARTICLES * sizeof(int), cudaMemcpyDeviceToHost);
    int sizeOfSL = sizeof(h_spatialLookup) / sizeof(h_spatialLookup[0]);
    qsort(h_spatialLookup, sizeOfSL, sizeof(float), compare);
    cudaMemcpy(d_spatialLookup, h_spatialLookup, sizeof(int), cudaMemcpyHostToDevice);

    updateSpacialLookup_step2<<<numBlocks, blockSize>>>(d_particles, d_spatialLookup, d_startIndex, d_NUM_PARTICLES);

    dim3 blockSize2(512, 512); 
    dim3 gridSize2((*h_NUM_PARTICLES + blockSize2.x - 1) / blockSize2.x, (*h_NUM_PARTICLES + blockSize2.y - 1) / blockSize2.y);

    calculateDensities<<<gridSize2, blockSize2>>>(d_densities, d_particles, d_NUM_PARTICLES);
    
    cudaDeviceSynchronize();

    cudaMemcpy(h_particles, d_particles, *h_NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_densities, d_densities, *h_NUM_PARTICLES * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_startIndex, d_startIndex, *h_NUM_PARTICLES * sizeof(int), cudaMemcpyDeviceToHost);


    cudaFree(d_particles);
    cudaFree(d_densities);
    cudaFree(d_NUM_PARTICLES);
    cudaFree(d_dt);
}