#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call){                                                                    \
    const cudaError_t error = call;                                                     \
    if (error != cudaSuccess){                                                          \
        printf("Error#: %s:%d \n", __FILE__, __LINE__);                                    \
        printf("\t code:%d, reason: %s\n", error, cudaGetErrorString(error));              \
        exit(1);                                                                        \
    }                                                                                   \
}   

void initialInt(int *ip, int size){
    for(int i=0; i<size; i++){
        ip[i] = i;
    }
}

void printMatrix(int *C, const int nx, const int ny){
    int *ic = C;
    printf("\n Matrix: (%d.%d) \n", nx, ny);
    for(int iy=0; iy<ny; iy++){
        for(int ix=0; ix<nx; ix++){
            printf("%3d", ic[ix]);
        }
        ic += nx;
        printf("\n");
    }
    printf("\n");
}

__global__ void printThreadIndex(int *A, const int nx, const int ny){
    int ix = threadIdx.x + blockIdx.x * blockDim.x; 
    int iy = threadIdx.y + blockIdx.y * blockDim.y;    

    unsigned int idx = iy * nx + ix;    
    printf("Thread_id (%d,%d) Block_id (%d,%d) Coordinate (%d,%d) " 
        "global index %2d ival %2d\n", threadIdx.x, threadIdx.y, blockIdx.x,
        blockIdx.y, ix, iy, idx, A[idx]);
}

int main(int argc, char **argv){
    printf("%s Starting ... \n", argv[0]);

    // Get device Info
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // Set Matrix Dim 
    int nx = 8;
    int ny = 6;
    int nxy = nx*ny;
    int nBytes = nxy * sizeof(float);

    // Malloc host mem
    int *h_A;
    h_A = (int *)malloc(nBytes);

    // Initialise host matrix with int
    initialInt(h_A, nxy);
    printMatrix(h_A, nx, ny);

    // Malloc device memory
    int *d_MatA;
    cudaMalloc((void **)&d_MatA, nBytes);

    // Transfer data from host to data
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);

    // Setup execution config 
    dim3 block(4,2);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y );

    // Invoke Kernal
    printThreadIndex <<<grid, block >>>(d_MatA, nx, ny);
    cudaDeviceSynchronize();

    // Free host and device Memory
    cudaFree(d_MatA);
    free(h_A);

    // Reset device
    cudaDeviceReset();

    return 0;

}