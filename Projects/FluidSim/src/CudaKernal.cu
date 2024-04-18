#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {
    #include "CudaKernal.h"
}

__global__ void kernel() {
    printf("Hello from CUDA kernel!\n");
}

void cuda_kernel() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
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

        particles[idx].x += particles[idx].dx * *dt;
        particles[idx].y -= particles[idx].dy * *dt;

        applyGravity(particles, idx);
        resolveCollisions(particles, idx);
    }
}

void __updateParticle(Particle *h_particles, float *h_dt, int *h_NUM_PARTICLES){
    // Allocate memory for the array on the GPU
    Particle *d_particles;
    int *d_NUM_PARTICLES;
    float *d_dt;
    cudaMalloc((void**)&d_particles, *h_NUM_PARTICLES * sizeof(Particle));
    cudaMalloc((void**)&d_NUM_PARTICLES, sizeof(int));
    cudaMalloc((void**)&d_dt, sizeof(float));

    cudaMemcpy(d_particles, h_particles, *h_NUM_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_NUM_PARTICLES, h_NUM_PARTICLES, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dt, h_dt, sizeof(float), cudaMemcpyHostToDevice);



    int blockSize = 256;
    int numBlocks = (*h_NUM_PARTICLES + blockSize - 1) / blockSize;
    cuda_updateParticle<<<numBlocks, blockSize>>>(d_particles, d_dt, d_NUM_PARTICLES);
    cudaDeviceSynchronize();

    cudaMemcpy(h_particles, d_particles, *h_NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);

    cudaFree(d_particles);
    cudaFree(d_NUM_PARTICLES);
    cudaFree(d_dt);
}