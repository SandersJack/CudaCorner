#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>

extern "C" {
    #include "NeuralNetwork.h"
}

__global__ void linearForwardProp(float* A, float* Z, ParametersLinear *params, int *num_images, int *num_rows, int *num_cols){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int A_x_dim = 60000;
    int A_y_dim = 784;

    int W_x_dim = 10;
    int W_y_dim = A_y_dim;

    int Z_x_dim = A_x_dim;
    int Z_y_dim = W_y_dim;

    float Z_value = 0;

    if(idx < Z_x_dim && idy < Z_y_dim){
        for(int t=0; t< W_x_dim; t++){
            Z_value += A[idx*A_y_dim + idy] * params->W[t * W_y_dim + idy] + params->B[t];
        }
        
        Z[idx * Z_y_dim + idy] = Z_value;
    }
}

__global__ void linearUpdateWeight(float* A, float* dZ, ParametersLinear *params){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int A_x_dim = 60000;
    int A_y_dim = 784;

    int W_x_dim = 10;
    int W_y_dim = A_y_dim;

    int dZ_x_dim = 10;
    int dZ_y_dim = A_y_dim;

    float dW_value = 0.0f;

    float learning_rate = 0.01;

    if(idx < W_x_dim && idy < W_y_dim){
        for(int i=0; i<dZ_x_dim; i++){
            dW_value += dZ[idx*W_y_dim + i] * A[idy * A_y_dim + i];
        }

        params->W[idx * W_y_dim + idy] = params->W[idx * W_y_dim + idy] - learning_rate * dW_value/A_x_dim;
    }
}

__global__ void linearUpdateBias(float *dZ, ParametersLinear *params){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int dZ_x_dim = 10;
    int dZ_y_dim = 60000;

    float learning_rate = 0.01;

    if(idx < dZ_x_dim * dZ_y_dim){
        int dZ_x = idx / dZ_x_dim;
        int dZ_y = idx % dZ_x_dim;
        atomicAdd(&params->B[dZ_y], -learning_rate * (dZ[dZ_x * dZ_y_dim + dZ_y] / dZ_y_dim));
    }
}

void ForwardProp(float *d_A, ParametersLinear *d_params, int *d_numImages, int *d_numRows, int *d_numCols, int *h_numImages, int *h_numRows, int *h_numCols){
    float *d_Z;
    // Z (A.x, W.y)
    cudaMalloc((void**)&d_Z, *h_numImages * *h_numRows * *h_numCols * sizeof(float));

    cudaError_t cudaError;

    dim3 block_size(8,8);
    dim3 num_of_blocks((*h_numImages+block_size.x-1)/block_size.x,(*h_numRows * *h_numCols+block_size.y-1)/block_size.y);

    linearForwardProp<<<num_of_blocks, block_size>>>(d_A, d_Z, d_params, d_numImages, d_numRows, d_numCols);
    cudaDeviceSynchronize();

    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("Kernel launch error (linearForwardProp): %s\n", cudaGetErrorString(cudaError));
        // Handle error appropriately
        exit(0);
    }
    
}

void BackProp(){}

void NeuralNetwork(float *h_data, int *h_numImages, int *h_numRows, int *h_numCols){

    ParametersLinear* h_params = (ParametersLinear*)malloc(sizeof(ParametersLinear));

    float *d_data;
    int *d_numImages, *d_numRows, *d_numCols;
    ParametersLinear *d_params;

    cudaMalloc((void**)&d_data, *h_numImages* *h_numRows * *h_numCols * sizeof(float));

    cudaMalloc((void**)&d_numImages, sizeof(int));
    cudaMalloc((void**)&d_numRows, sizeof(int));
    cudaMalloc((void**)&d_numCols, sizeof(int));

    cudaMalloc((void**)&d_params, sizeof(ParametersLinear));

    cudaMemcpy(d_data, h_data, *h_numImages* *h_numRows * *h_numCols * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_numImages, h_numImages, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numRows, h_numRows, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numCols, h_numCols, sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_params, h_params, sizeof(ParametersLinear), cudaMemcpyHostToDevice);

    // Testing with one forward prop
    ForwardProp(d_data, d_params, d_numImages, d_numRows, d_numCols, h_numImages, h_numRows, h_numCols);
}