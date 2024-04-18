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