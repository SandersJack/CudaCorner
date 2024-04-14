#include <cuda_runtime.h>
#include <stdio.h>

__global__ void checkIndex(void){
    printf("ThreadIdx: (%d, %d, %d) BlockIdx: (%d, %d, %d) BlockDim: (%d, %d, %d) GridDim (%d, %d, %d) \n", 
            threadIdx.x, threadIdx.y, threadIdx.z,
            blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z,
            gridDim.x,gridDim.y,gridDim.z);
}

int main(int argc, char **argv){
    // Define total data element
    int nElem = 6;

    // Define grid and block stucture
    dim3 block(3);
    dim3 grid(( nElem + block.x-1) /  block.x);

    // Check grid and block dimension from host side
    printf("grid.x %d grid.y %d grid.z %d\n",grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n",block.x, block.y, block.z);

    checkIndex <<<grid, block>>>();

    // Reset device
    cudaDeviceReset();

    return 0;

}