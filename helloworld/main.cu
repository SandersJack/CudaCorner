#include <stdio.h>

__global__ void helloFromGPU(void){
    int thread = threadIdx.x;
    printf("Hello World from GPU thread %i \n", thread);
}

int main(void) {

    printf("Hello World from CPU. \n");

    helloFromGPU <<<1, 10>>>();
    //cudaDeviceReset();
    cudaDeviceSynchronize();
    return 0;
}