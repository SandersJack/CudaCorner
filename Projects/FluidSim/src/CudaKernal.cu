#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>

extern "C" {
    #include "CudaKernal.h"
}

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
    particles[id].dy -= 0.1;
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

__global__ void cuda_updateParticlePred(Particle *particles, float *dt, int *NUM_PARTICLES){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < *NUM_PARTICLES){

        particles[idx].pred_x = particles[idx].x + particles[idx].dx * *dt;
        particles[idx].pred_y = particles[idx].y - particles[idx].dy * *dt;

    }
}

__global__ void cuda_updateParticle(Particle *particles, float *densities, float *dt, int *NUM_PARTICLES, FloatPair *pressureForce){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < *NUM_PARTICLES){

        float pressureAccel_X = pressureForce[idx].first;
        float pressureAccel_Y = pressureForce[idx].second;

        //printf("Pressure Accel %f %f \n", pressureForce[idx].first, pressureForce[idx].second);
        //printf("Density %f \n", densities[idx]);

        atomicAdd(&particles[idx].dx , pressureAccel_X * *dt);
        atomicAdd(&particles[idx].dy , pressureAccel_Y * *dt);

        //applyGravity(particles, idx);
        resolveCollisions(particles, idx);

        atomicAdd(&particles[idx].x , particles[idx].dx * *dt);
        atomicAdd(&particles[idx].y , -particles[idx].dy * *dt);

    }
}

static __device__ uIntPair posToCellCoord(Particle particle, float radius){
    uIntPair pair;
    pair.first = (uint)(particle.pred_x / radius);
    pair.second = (uint)(particle.pred_y / radius);
    return pair;
}

static __device__ uint hashCell(int cellX, int cellY){
    uint a = cellX * 15823;
    uint b = cellY * 9737333;
    //printf("Val %i %i %i\n", cellX, cellY, a+b);
    return a + b;
}

static __device__ uint getKeyFromHash(int *NUM_PARTICLES, uint hash){
    int val =  hash % *NUM_PARTICLES;
    //printf("Val %i %i %i\n", val, hash, *NUM_PARTICLES);
    return val;
}


void __global__ calculateDensities(float* densities, Particle *particles, int *NUM_PARTICLES, int* spatialLookup, SpacialIndex *spacialIndexs, IntPair *offsets, 
        float *smoothingRadius, float *MASS){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < *NUM_PARTICLES){
        uIntPair origin_cell = posToCellCoord(particles[idx], *smoothingRadius);
        float density = 0;

        float sqrRadius = *smoothingRadius * *smoothingRadius;

        for(int i=0; i<9; i++){
            
            uint hash = hashCell(origin_cell.first + offsets[i].first, origin_cell.second + offsets[i].second);
		    uint key = getKeyFromHash(NUM_PARTICLES, hash);

		    uint currIndex = spatialLookup[key];
            

            while (currIndex < *NUM_PARTICLES)
		    {
                SpacialIndex indexData = spacialIndexs[currIndex];
                currIndex++;
                
                //printf("Data %i %i \n", indexData.key, key);
                if(indexData.key == -99 || indexData.hash == -99 || indexData.key == -99) continue;

                if(indexData.key != key) break;
                if(indexData.hash != hash) continue;

                int n_index = indexData.index;

                if(n_index > *NUM_PARTICLES) continue;

                float dist_x = particles[n_index].pred_x - particles[idx].pred_x;
                float dist_y = particles[n_index].pred_y - particles[idx].pred_y;

                float sqrdist = (dist_x * dist_x + dist_y * dist_y);

 
                if(sqrdist > sqrRadius) continue;

                float dist = sqrt(sqrdist);
                float influence = smoothingKernal_new(*smoothingRadius, dist);
                density += *MASS * influence;
                
            }
        }
        densities[idx] = density;
    }
}
__global__ void updateSpacialLookup_step1(Particle *particles, int *spatialLookup, SpacialIndex *spacialIndexs, int *NUM_PARTICLES, float *smoothingRadius){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < *NUM_PARTICLES){
        uIntPair cell = posToCellCoord(particles[idx], *smoothingRadius);
        uint cellKey = getKeyFromHash(NUM_PARTICLES, hashCell(cell.first, cell.second));
        spatialLookup[idx] = cellKey;

        spacialIndexs[idx].index = -99; spacialIndexs[idx].hash = -99; spacialIndexs[idx].key = -99;
    }
}

__global__ void updateSpacialLookup_step2(Particle *particles, int *spatialLookup, SpacialIndex *spacialIndexs, int *NUM_PARTICLES, float *smoothingRadius){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    
    if(idx < *NUM_PARTICLES){
        int key = spatialLookup[idx];
        int keyPrev = (idx == 0) ? -99 : spatialLookup[idx-1];

        if(key != keyPrev){
            uIntPair cell = posToCellCoord(particles[idx], *smoothingRadius);
            uint hash = hashCell(cell.first, cell.second);
            int cellKey = getKeyFromHash(NUM_PARTICLES, hash);
            //printf("Indexs %i %i %i \n", idx, hash, cellKey);
            spacialIndexs[idx].index = idx; spacialIndexs[idx].hash = hash; spacialIndexs[idx].key = cellKey; 

        }
    }
}


__device__ float convertDensityToPressure(float density, float PRESSURE_MULT){

    float TARGET_DENSITY = 0.0001;

    float densityError = density / TARGET_DENSITY;
    float diff = density - TARGET_DENSITY;
    if (diff < 0) densityError *= -1;

    float pressure =  densityError * PRESSURE_MULT;
    return pressure;
}


__device__ float GetRandomDir() {
    // Initialize curand state for the thread
    curandState localState;
    curand_init(clock64(), threadIdx.x, 0, &localState);

    // Generate a random floating-point value between -1 and 1
    return 2.0f * curand_uniform(&localState) - 1.0f;
}

__global__ void calculateDensityForces(float* densities, Particle *particles, int *NUM_PARTICLES, int* spatialLookup, 
    SpacialIndex *spacialIndexs, IntPair *offsets, FloatPair *pressureForces, float* smoothingRadius, float *MASS, 
    float *pressureMult){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if( idx < *NUM_PARTICLES){

        //printf("Density %f \n", densities[idx]);

        float pressure = convertDensityToPressure(densities[idx], *pressureMult);

        FloatPair __pressureForce; __pressureForce.first = 0.; __pressureForce.second = 0.;
        uIntPair origin_cell = posToCellCoord(particles[idx], *smoothingRadius);


        float sqrRadius = *smoothingRadius * *smoothingRadius;

        for(int i=0; i<9; i++){
            uint hash = hashCell(origin_cell.first + offsets[i].first, origin_cell.second + offsets[i].second);
		    uint key = getKeyFromHash(NUM_PARTICLES, hash);
            
		    uint currIndex = spatialLookup[key];
            while (currIndex < *NUM_PARTICLES)
		    {
                //printf("Current Index %i \n", currIndex);
                
                SpacialIndex indexData = spacialIndexs[currIndex];
                currIndex++;
                
                if(indexData.key == -99 || indexData.hash == -99 || indexData.key == -99) continue;

                if(indexData.key != key) break;
                if(indexData.hash != hash) continue;

                int n_index = indexData.index;

                if(n_index == idx) continue;

                if(n_index > *NUM_PARTICLES) continue;
                
                //printf("Index %i \n", n_index);

                float dist_X = particles[n_index].pred_x - particles[idx].pred_x;
                float dist_Y = particles[n_index].pred_y - particles[idx].pred_y;
                float sqDist = (dist_X*dist_X + dist_Y*dist_Y);

                if(sqDist > sqrRadius) continue;

                float dist = sqrt(sqDist);

                float dir_X = (dist <= 0) ? GetRandomDir(): dist_X / dist;
                float dir_Y = (dist <= 0) ? GetRandomDir() : dist_Y / dist;
                
                float n_density = densities[n_index];

                float neighbourPressure = convertDensityToPressure(n_density, *pressureMult);
                //printf(" %i Density A and B %i %f  %i %f \n", currIndex, n_index, densities[n_index], idx, densities[idx]);
                float sharedPressure = (pressure + neighbourPressure) * 0.5;

                float influence = smoothingKernalDerivative_new(*smoothingRadius, dist);

                __pressureForce.first += dir_X * sharedPressure * influence / n_density;
                __pressureForce.second -= dir_Y * sharedPressure * influence / n_density;
                

            }
        }

        pressureForces[idx].first = __pressureForce.first; pressureForces[idx].second = __pressureForce.second; 
    }
}

void __updateParticle(Particle *h_particles, float *h_dt, int *h_NUM_PARTICLES, float *h_densities, int *h_spatialLookup, SpacialIndex *h_spacialIndexs, 
        FloatPair *h_pressureForce, IntPair *h_offsets, float *h_smoothingRadius, float *h_mass, float *h_pressureMult){
    // Allocate memory for the array on the GPU
    Particle *d_particles;
    int *d_NUM_PARTICLES, *d_spatialLookup, *d_spatialLookup2;
    float *d_dt, *d_densities, *d_smoothingRadius, *d_mass, *d_pressureMult;

    FloatPair *d_pressureForce;
    IntPair *d_offsets;
    SpacialIndex *d_spacialIndexs;

    cudaError_t cudaError;

    cudaMalloc((void**)&d_particles, *h_NUM_PARTICLES * sizeof(Particle));
    cudaMalloc((void**)&d_NUM_PARTICLES, sizeof(int));

    cudaMalloc((void**)&d_spatialLookup, *h_NUM_PARTICLES * sizeof(int));
    cudaMalloc((void**)&d_spatialLookup2, *h_NUM_PARTICLES * sizeof(int));

    cudaMalloc((void**)&d_spacialIndexs, *h_NUM_PARTICLES * sizeof(SpacialIndex));
    
    cudaMalloc((void**)&d_pressureMult, sizeof(float));
    cudaMalloc((void**)&d_mass, sizeof(float));
    cudaMalloc((void**)&d_smoothingRadius, sizeof(float));
    cudaMalloc((void**)&d_dt, sizeof(float));
    cudaMalloc((void**)&d_densities, *h_NUM_PARTICLES * sizeof(float));
    cudaMalloc((void**)&d_pressureForce, *h_NUM_PARTICLES * sizeof(FloatPair));
    cudaMalloc((void**)&d_offsets, 9 * sizeof(IntPair));


    cudaMemcpy(d_particles, h_particles, *h_NUM_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_NUM_PARTICLES, h_NUM_PARTICLES, sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_spatialLookup, h_spatialLookup, *h_NUM_PARTICLES * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spacialIndexs, h_spacialIndexs, *h_NUM_PARTICLES * sizeof(SpacialIndex), cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_pressureMult, h_pressureMult, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, h_mass, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_smoothingRadius, h_smoothingRadius, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dt, h_dt, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_densities, h_densities, *h_NUM_PARTICLES * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pressureForce, h_pressureForce, *h_NUM_PARTICLES * sizeof(FloatPair), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets, 9 * sizeof(IntPair), cudaMemcpyHostToDevice);

    int blockSize = 512;
    int numBlocks = (*h_NUM_PARTICLES + blockSize - 1) / blockSize;

    cuda_updateParticlePred<<<numBlocks, blockSize>>>(d_particles, d_dt, d_NUM_PARTICLES);
    
    cudaDeviceSynchronize();

    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("Kernel launch error (cuda_updateParticlePred): %s\n", cudaGetErrorString(cudaError));
        // Handle error appropriately
        exit(0);
    }
    
    updateSpacialLookup_step1<<<numBlocks, blockSize>>>(d_particles, d_spatialLookup, d_spacialIndexs, d_NUM_PARTICLES, d_smoothingRadius);

    cudaDeviceSynchronize();

    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("Kernel launch error (updateSpacialLookup_step1): %s\n", cudaGetErrorString(cudaError));
        // Handle error appropriately
        exit(0);
    }
    
    
    cudaMemcpy(h_spatialLookup, d_spatialLookup, *h_NUM_PARTICLES * sizeof(int), cudaMemcpyDeviceToHost);

    int sizeOfSL = sizeof(h_spatialLookup) / sizeof(h_spatialLookup[0]);
    qsort(h_spatialLookup, sizeOfSL, sizeof(float), compare);

    cudaMemcpy(d_spatialLookup2, h_spatialLookup, *h_NUM_PARTICLES * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("Kernel launch error (copy): %s\n", cudaGetErrorString(cudaError));
        // Handle error appropriately
        exit(0);
    }
    
    updateSpacialLookup_step2<<<numBlocks, blockSize>>>(d_particles, d_spatialLookup2, d_spacialIndexs, d_NUM_PARTICLES, d_smoothingRadius);
    
    cudaDeviceSynchronize();

    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("Kernel launch error (updateSpacialLookup_step2): %s\n", cudaGetErrorString(cudaError));
        // Handle error appropriately
        exit(0);
    }

    calculateDensities<<<numBlocks, blockSize>>>(d_densities, d_particles, d_NUM_PARTICLES, d_spatialLookup, d_spacialIndexs, d_offsets, d_smoothingRadius, d_mass);

    cudaDeviceSynchronize();

    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("Kernel launch error (calculateDensities): %s\n", cudaGetErrorString(cudaError));
        // Handle error appropriately
        exit(0);
    }

    
    calculateDensityForces<<<numBlocks, blockSize>>>(d_densities, d_particles, d_NUM_PARTICLES, d_spatialLookup, d_spacialIndexs, d_offsets, 
            d_pressureForce, d_smoothingRadius, d_mass, d_pressureMult);
    cudaDeviceSynchronize();
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("Kernel launch error (calculateDensityForces): %s\n", cudaGetErrorString(cudaError));
        // Handle error appropriately
        exit(0);
    }
    

    

    cuda_updateParticle<<<numBlocks, blockSize>>>(d_particles, d_densities, d_dt, d_NUM_PARTICLES, d_pressureForce);

    cudaDeviceSynchronize();

    cudaMemcpy(h_particles, d_particles, *h_NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_densities, d_densities, *h_NUM_PARTICLES * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_spacialIndexs, d_spacialIndexs, *h_NUM_PARTICLES * sizeof(SpacialIndex), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_spatialLookup, d_spatialLookup, *h_NUM_PARTICLES * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pressureForce, d_pressureForce, *h_NUM_PARTICLES * sizeof(FloatPair), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    float avgDen = 0;
    for(int i=0; i<*h_NUM_PARTICLES; i++){
        avgDen += h_densities[i];
        //printf("Particles v %f %f \n", h_densities[i], h_pressureForce[i].second);
    }
    printf("Average Density %f \n", avgDen / *h_NUM_PARTICLES);

    cudaFree(d_particles);
    cudaFree(d_NUM_PARTICLES);
    cudaFree(d_spatialLookup);
    cudaFree(d_spatialLookup2);
    cudaFree(d_dt);
    cudaFree(d_densities);
    cudaFree(d_pressureForce);
    cudaFree(d_offsets);
    cudaFree(d_spacialIndexs);
}