#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

#define CHECK(call){                                                                    \
    const cudaError_t error = call;                                                     \
    if (error != cudaSuccess){                                                          \
        printf("Error#: %s:%d \n", __FILE__, __LINE__);                                    \
        printf("\t code:%d, reason: %s\n", error, cudaGetErrorString(error));              \
        exit(1);                                                                        \
    }                                                                                   \
}   

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void printMatrix(float *C, const int nx, const int ny){
    float *ic = C;
    printf("\n Matrix: (%d.%d) \n", nx, ny);
    for(int iy=0; iy<ny; iy++){
        for(int ix=0; ix<nx; ix++){
            printf("%3f ", ic[ix]);
        }
        ic += nx;
        printf("\n");
    }
    printf("\n");
}

void initialData(float *ip,int size) {
    // generate different seed for random number
    time_t t;
    srand((unsigned) time(&t));
    for (int i=0; i<size; i++) {
        ip[i] = (float)( rand() & 0xFF )/10.0f;
    }
}

void checkResult(float *hostRef, float *gpuRef, const int N) {
    double epsilon = 1.0e-9;
    bool match = 1;

    for(int i=0; i<N; i++){
        if(abs(hostRef[i] - gpuRef[i]) > epsilon){
            match = 0;
            printf("Arrays do not match! \n");
            printf("Host %5.2f GPU %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }

    if(match) printf("Arrays match! \n\n");
}

void transposeOnCPU(float *A, float *C, const int nx, const int ny){
    float *ia = A;
    float *ic = C;

    for(int ix=0; ix < nx; ix++){
        for(int iy=0; iy<ny; iy++){
            ic[iy * ny + ix] = ia[ix * ny + iy];
        }
    }
}

__global__ void transposeOnGPU(float *A, float *C, const int nx, const int ny){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix < nx && iy < ny){
        C[iy * ny + ix] = A[ix * ny + iy];
    }
}


int main(int argc, char **argv){
    printf("%s Starting ... \n", argv[0]);

    // Get device Info
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nx = 1 << 14;
    int ny = 1<< 14;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: (%d, %d) \n", nx, ny);

    float *h_A, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // CPU
    double iStart = cpuSecond();
    initialData(h_A, nxy);
    //printMatrix(h_A, nx, ny);
    double iElaps = cpuSecond() - iStart;
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);
    iStart = cpuSecond();
    transposeOnCPU(h_A, hostRef, nx, ny);
    iElaps = cpuSecond() - iStart;
    printf("ONCPU elapsed %f sec\n", iElaps);
    //printMatrix(hostRef, nx, ny);

    // Intialise data on Device
    float *d_MatA, *d_MatC;
    cudaMalloc((void **)&d_MatA, nBytes);
    cudaMalloc((void **)&d_MatC, nBytes);

    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);

    // Invoke Kernal 
    int dimx = 32;
    int dimy = 16;
    dim3 block(dimx, dimy);
    dim3 grid(((nx + block.x-1)/block.x), ((ny + block.y-1) / block.y));

    iStart = cpuSecond();
    transposeOnGPU <<< grid, block >>>(d_MatA, d_MatC, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;

    printf("OnGPU <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x,
            grid.y, block.x, block.y, iElaps);

    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);
    
    //printMatrix(gpuRef, nx, ny);
    checkResult(hostRef, gpuRef, nxy);

    cudaFree(d_MatA);
    cudaFree(d_MatC);

    free(h_A);
    free(hostRef);
    free(gpuRef);
    // reset device
    cudaDeviceReset();

    return 0;
}
