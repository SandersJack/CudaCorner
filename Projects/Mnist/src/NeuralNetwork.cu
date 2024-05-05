#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>

#define MAX_SAFE_VALUE_FOR_EXP 50

extern "C" {
    #include "NeuralNetwork.h"
}

__global__ void findMax(float* Z, float* maxZ, int Z_x_dim, int Z_y_dim) {
    int idx =  blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < Z_x_dim) {
        float max_val = 1;
        for (int i = 0; i < Z_y_dim; i++) {
            float val = Z[idx * Z_y_dim + i];
            if (val > max_val) {
                max_val = val;
            }
            //if(idx == 550)
            //    printf("maxVals %i %f \n", i, Z[idx * Z_y_dim + i]);
        }
        maxZ[idx] = max_val;
    }
}

__global__ void sumExp(float *Z2, float *sum_exp, float *maxZ, int Z_x_dim, int Z_y_dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float sum_val = 0.;

    if(idx < Z_x_dim){
        //printf("MaxZ %f \n", maxZ[idx]);
        for(int i=0; i<Z_y_dim; i++){
            Z2[idx*Z_y_dim + i] -= maxZ[idx];
            sum_val += exp(Z2[idx*Z_y_dim + i]);
            //if(idx == 550)
            //    printf(" sumExp %f %f %f \n", Z2[idx*Z_y_dim + i], exp(Z2[idx*Z_y_dim + i]), maxZ[idx]);
        }
        sum_exp[idx] = sum_val;
    }
}

__global__ void softMax(float *Z2, float *A2, float *sum_exp, 
    int Z_x_dim, int Z_y_dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < Z_x_dim){
        for(int i=0; i<Z_y_dim; i++){
            A2[idx * Z_y_dim + i] = exp(Z2[idx * Z_y_dim + i]) / sum_exp[idx];
        }
    }
}

__global__ void reLUBack(float* Z1, float* dA1, int Z_x_dim, int Z_y_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < Z_x_dim * Z_y_dim) {
        if(Z1[idx] > 0){
            dA1[idx] *= 1;
        } else {
            dA1[idx] *= 0;
        }
    }
}

__global__ void reLUForward(float *Z1, float *A1, int Z_x_dim, int Z_y_dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < Z_x_dim * Z_y_dim){
        if(Z1[idx] > 0){
            A1[idx] = Z1[idx];
        } else {
            A1[idx] = 0;
        }
    }
}

__global__ void transposeWeights(ParametersLinear *W, ParametersLinear *W_T, int A_x_dim, int A_y_dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if(idx < A_x_dim && idy < A_y_dim){
        W_T->W[idy * A_x_dim + idx] = W->W[idx * A_y_dim + idy];
        W_T->B[idx] = W->B[idx];
    }
}

__global__ void transposeMatrix(float *A, float *A_T, int A_x_dim, int A_y_dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if(idx < A_x_dim && idy < A_y_dim){
        A_T[idy * A_x_dim + idx] = A[idx * A_y_dim + idy];
        //if(idy < 784 && idx < 1)
            //printf("Transpose: %i %i %f \n", idy * A_x_dim + idx, idx * A_y_dim + idy, A_T[idy * A_x_dim + idx]);
    }
}

__global__ void linearForwardProp(float* A, float* Z, ParametersLinear *params, int *num_rows, int *num_cols,
    int Z_x_dim, int Z_y_dim, int W_x_dim, int W_y_dim, int A_x_dim, int A_y_dim, int printout){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    float Z_value = 0;
    
    if(idx < Z_x_dim && idy < Z_y_dim){
        for(int t=0; t< W_y_dim; t++){
            ///                             // FP1          // FP2
            int W_idx = idx * W_y_dim + t; // W 10 x 784    // W 10 x 10
            int A_idx = t * A_y_dim + idy; // A 784 x 1000  // A 10 x 1000
            Z_value += params->W[W_idx] * A[A_idx];
            //printf("%i %i %i %i %i Z: %f %f \n", idx, idy , t, A_idx, W_idx,  A[A_idx] , params->W[W_idx]);
        }                                                   // FP1      // FP2
        Z[idx * Z_y_dim + idy] = Z_value + params->B[idy]; // 10 x 1000 //10 x 1000
    }
}

__global__ void linearBackProp(float *dZ2, float *dZ1, ParametersLinear *params, int dZ_x_dim, int dZ_y_dim, int W_x_dim, int W_y_dim, int dZ2_x_dim, int dZ2_y_dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    float dZ1_value = 0.0f;

    if(idx < dZ_x_dim && idy < dZ_y_dim){
        for(int i=0; i<W_x_dim; i++){
            int W_idx = idx * W_y_dim + i; // W.T 10 x 10
            int dZ2_idx = i * dZ2_y_dim + idy; // dZ2 10 x 1000

            dZ1_value += params->W[W_idx] * dZ2[dZ2_idx];
            //printf("linear back %i %i %f %f \n", W_idx, dZ2_idx, params->W[W_idx], dZ2[dZ2_idx]);
        }
        dZ1[idx*dZ_y_dim+idy] = dZ1_value; // 10 x 1000
    }
}

__global__ void linearUpdateWeight(float* A, float* dZ, ParametersLinear *params, int W_x_dim, int W_y_dim, int A_x_dim, int A_y_dim, int dZ_x_dim, int dZ_y_dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    float dW_value = 0.0f;

    float learning_rate = 0.1;

    if(idx < W_x_dim && idy < W_y_dim){
        for(int i=0; i<dZ_x_dim; i++){
            int dZ_idx = idx * dZ_y_dim + idx ; // Dz eg dz1 10 x 1000 
            int A_idx = i * A_y_dim + idy; // A.T eg X 1000 x 784

            dW_value += dZ[dZ_idx] * A[A_idx];
        }

        //printf("Update W: %i %i %f \n", idx * W_y_dim + idy, idx, dW_value/A_x_dim);
        params->W[idx * W_y_dim + idy] = params->W[idx * W_y_dim + idy] - learning_rate * dW_value/A_x_dim;

    }
}

__global__ void calcDiffBias(float* dZ, float* dZ_sum, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(dZ_sum, dZ[idx]);
    }
}

__global__ void linearUpdateBias(float *dZ_sum, ParametersLinear *params, int dZ_x_dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    

    float learning_rate = 0.1;

    if(idx < dZ_x_dim){
        atomicAdd(&params->B[idx], -learning_rate * (*dZ_sum / (60000.)));
        
        //printf("Update B: %i %f \n", idx, *dZ_sum);
    }
}

void ForwardProp(float *d_X, ParametersLinear *d_params1, ParametersLinear *d_params2, int *d_numRows, 
    int *d_numCols, int *h_numImages, int *h_numRows, int *h_numCols,
    float *d_Z1, float *d_A1, float *d_Z2, float *d_A2){

    float *d_sum_exp; 
    cudaMalloc((void**)&d_sum_exp, *h_numImages * 10 * sizeof(float));

    float* d_maxZ;
    cudaMalloc(&d_maxZ, *h_numImages * sizeof(float));

    int matrixSize = *h_numImages * 784;
    //printf("Start Forward \n");
    cudaError_t cudaError;

    int batchSize = 1000;

    int numBatches = (*h_numImages) / batchSize;

    int batcharraysize = batchSize * 784;

    dim3 block_size(16,16);
    dim3 num_of_blocks((batchSize+block_size.x-1)/block_size.x, (*h_numRows * *h_numCols+block_size.y-1)/block_size.y);

    dim3 num_of_blocksFP1((10+block_size.x-1)/block_size.x, (batchSize+block_size.y-1)/block_size.y);

    dim3 num_of_blocksFP2((10+block_size.x-1)/block_size.x, (batchSize+block_size.y-1)/block_size.y);

    int singleDimblockSize = 512;
    int singleDimnumBlocks = (batcharraysize + singleDimblockSize - 1) / singleDimblockSize;
    int singleDimnumBlocks10 = (batchSize * 10 + singleDimblockSize - 1) / singleDimblockSize;
    int singleDimnumBlocks2 = (batchSize + singleDimblockSize - 1) / singleDimblockSize;

    //printf("Matrix Size %i \n", matrixSize);
    //printf("Num block x threads %i \n", singleDimblockSize * singleDimnumBlocks);
    //printf("Num Threads %i \n", singleDimnumBlocks);

    for (int i = 0; i < numBatches; i++) {
        //printf("Batch %i Start \n", i);
        // Calculate the start and end indices for this batch
        int startIdx = i * batchSize;
        int endIdx = min(startIdx + batchSize, *h_numImages);
        // 47040000
        // 10976000
        // Calculate the size of this batch
        int batchMatrixSize = (endIdx - startIdx) * 784;
        //printf("%i \n", startIdx * 784);
        /// First Linear Layer
        linearForwardProp<<<num_of_blocksFP1, block_size>>>(d_X, d_Z1, d_params1, d_numRows, d_numCols,
            10, batchSize, 10 , 784, 784, 1000, 0);
        cudaDeviceSynchronize();
        //exit(0);
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Kernel launch error (linearForwardProp1): %s\n", cudaGetErrorString(cudaError));
            // Handle error appropriately
            exit(0);
        }
        
        /// Relu Layer
        reLUForward<<<singleDimnumBlocks, singleDimblockSize>>>(d_Z1, d_A1, batchSize, 10);
        cudaDeviceSynchronize();

        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Kernel launch error (reLUForward): %s\n", cudaGetErrorString(cudaError));
            // Handle error appropriately
            exit(0);
        }
        /// Second Linear Layer
        linearForwardProp<<<num_of_blocksFP2, block_size>>>(d_A1, d_Z2, d_params2, d_numRows, d_numCols, 
            10, batchSize, 10, 10, 10, 1000, 0);
        cudaDeviceSynchronize();
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Kernel launch error (linearForwardProp2): %s\n", cudaGetErrorString(cudaError));
            // Handle error appropriately
            exit(0);
        }
        findMax<<<singleDimnumBlocks10, singleDimblockSize>>>(d_Z2, d_maxZ, 1000, 10);
        cudaDeviceSynchronize();
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Kernel launch error (findMax): %s\n", cudaGetErrorString(cudaError));
            // Handle error appropriately
            exit(0);
        }
        /// Sum the exponetial 
        sumExp<<<singleDimnumBlocks10, singleDimblockSize>>>(d_Z2, d_sum_exp, d_maxZ,  1000, 10);
        cudaDeviceSynchronize();
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Kernel launch error (sumExp): %s\n", cudaGetErrorString(cudaError));
            // Handle error appropriately
            exit(0);
        }
        /// Conduct the SoftMax
        softMax<<<singleDimnumBlocks10, singleDimblockSize>>>(d_Z2, d_A2, d_sum_exp, 
        1000, 10);
        cudaDeviceSynchronize();
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Kernel launch error (softMax): %s\n", cudaGetErrorString(cudaError));
            // Handle error appropriately
            exit(0);
        }
        /// Pointer arith
        d_X += batchSize * 784;
        d_Z1 += batchSize * 10;
        d_A1 += batchSize * 10;
        d_Z2 += batchSize * 10;
        d_A2 += batchSize * 10;
    }
}

__global__ void startBackProp(float *d_Z2, float *d_A2, unsigned char *d_one_hot_Y, int Z_x_dim, int Z_y_dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < Z_x_dim * Z_y_dim){
        d_Z2[idx] = 2 * (d_A2[idx] - d_one_hot_Y[idx]);
    }
}

void BackProp(float *d_Z1, float *d_A1, float *d_A2, ParametersLinear *d_params1, ParametersLinear *d_params2, unsigned char *d_one_hot_Y, 
    float *d_data, float* d_dZ2, float *d_dZ1, int *h_numImages, int *h_numRows, int *h_numCols){
    
    //printf("Start Back \n");
    cudaError_t cudaError;

    int batchSize = 1000;

    int numBatches = (*h_numImages) / batchSize;

    int batcharraysize = batchSize * 784;
    int batcharraysize10 = batchSize * 10;

    dim3 block_size(16,16);
    dim3 num_of_blocks((batchSize+block_size.x-1)/block_size.x,(*h_numRows * *h_numCols+block_size.y-1)/block_size.y);
    dim3 num_of_blocks1010((10+block_size.x-1)/block_size.x, (10+block_size.y-1)/block_size.y);
    dim3 num_of_blocks_back((10 + block_size.x-1)/block_size.x, (batchSize+block_size.y-1)/block_size.y);

    int singleDimblockSize = 512;
    int singleDimnumBlocks784 = (batcharraysize + singleDimblockSize - 1) / singleDimblockSize;
    int singleDimnumBlocks10 = (batcharraysize10 + singleDimblockSize - 1) / singleDimblockSize;
    int _singleDimnumBlocks10 = (10 + singleDimblockSize - 1) / singleDimblockSize;
    int singleDimnumBlocks10x10 = (10 * 10 + singleDimblockSize - 1) / singleDimblockSize;
    int singleDimnumBlocks = (batchSize + singleDimblockSize - 1) / singleDimblockSize;

    //// Transpose varius matrices
    float * d_A1_T;
    cudaMalloc((void**)&d_A1_T, *h_numImages * 10 * sizeof(float));

    ParametersLinear *d_params2_WT;
    cudaMalloc((void**)&d_params2_WT, sizeof(ParametersLinear));
  
    float* d_W2_T;
    cudaMalloc((void**)&d_W2_T, 10 * 10 * sizeof(float));
    cudaMemcpy(&(d_params2_WT->W), &d_W2_T, sizeof(float*), cudaMemcpyHostToDevice);

    float* d_B_T;
    cudaMalloc((void**)&d_B_T, 10 * 1 * sizeof(float));
    cudaMemcpy(&(d_params2_WT->B), &d_B_T, sizeof(float*), cudaMemcpyHostToDevice);


    //cudaMemcpy(&(d_params2_WT->B), &(d_params2->B), sizeof(float*), cudaMemcpyHostToDevice);
    // Batches
    for (int i = 0; i < *h_numImages; i += batchSize) {
        int batch_size = min(1000, *h_numImages - i);
        /// A1
        dim3 num_of_blocks_T((batch_size + block_size.x - 1) / block_size.x, (10 + block_size.y - 1) / block_size.y);
        transposeMatrix<<<num_of_blocks_T, block_size>>>(d_A1 + i * 10, d_A1_T + i * 10, batch_size, 10);
        cudaDeviceSynchronize();
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Kernel launch error (transposeMatrixA1): %s\n", cudaGetErrorString(cudaError));
            // Handle error appropriately
            exit(0);
        }
    }
    
    // Weights 
    dim3 num_of_blocks_T((10 + block_size.x - 1) / block_size.x, (10 + block_size.y - 1) / block_size.y);
    transposeWeights<<<num_of_blocks_T, block_size>>>(d_params2, d_params2_WT, 10, 10);
    cudaDeviceSynchronize();
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("Kernel launch error (transposeMatrixW2): %s\n", cudaGetErrorString(cudaError));
        // Handle error appropriately
        exit(0);
    }

    float *d_dZ2_sum;
    cudaMalloc((void**)&d_dZ2_sum, sizeof(float));

    float *d_dZ1_sum;
    cudaMalloc((void**)&d_dZ1_sum, sizeof(float));

    for (int i = 0; i < numBatches; i++) {
        //printf("Batch %i Start \n", i);
        // Calculate the start and end indices for this batch
        int startIdx = i * batchSize;
        int endIdx = min(startIdx + batchSize, *h_numImages);

        // Calculate the size of this batch
        int batchMatrixSize = (endIdx - startIdx) * 784;
        //printf("%i \n", startIdx * 10);

        startBackProp<<<singleDimnumBlocks10, singleDimblockSize>>>(d_dZ2, d_A2, d_one_hot_Y, batchSize, 10);
        cudaDeviceSynchronize();
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Kernel launch error (startBackProp): %s\n", cudaGetErrorString(cudaError));
            // Handle error appropriately
            exit(0);
        }
        linearBackProp<<<num_of_blocks_back, block_size>>>(d_dZ2, d_dZ1, d_params2_WT, 10, batchSize, 10, 10 , 10, batchSize);
        cudaDeviceSynchronize();
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Kernel launch error (linearBackProp): %s\n", cudaGetErrorString(cudaError));
            // Handle error appropriately
            exit(0);
        }
        reLUBack<<<singleDimnumBlocks10, singleDimblockSize>>>(d_Z1, d_dZ1, batchSize, 10);
        cudaDeviceSynchronize();
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Kernel launch error (reLUBack): %s\n", cudaGetErrorString(cudaError));
            // Handle error appropriately
            exit(0);
        }
        // Linear 2
        linearUpdateWeight<<<num_of_blocks1010, block_size>>>(d_A1_T, d_dZ2, d_params2, 10, 10, batchSize, 10, 10, batchSize);
        cudaDeviceSynchronize();
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Kernel launch error (linearUpdateWeight2): %s\n", cudaGetErrorString(cudaError));
            // Handle error appropriately
            exit(0);
        }
        
        calcDiffBias<<<singleDimnumBlocks10, singleDimblockSize>>>(d_dZ2, d_dZ2_sum, batchSize);
        cudaDeviceSynchronize();
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Kernel launch error (calcDiffBias2): %s\n", cudaGetErrorString(cudaError));
            // Handle error appropriately
            exit(0);
        }
        // Linear 1        
        linearUpdateWeight<<<num_of_blocks, block_size>>>(d_data, d_dZ1, d_params1, 784, 10, 784, batchSize, batchSize, 10);
        cudaDeviceSynchronize();
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Kernel launch error (linearUpdateWeight1): %s\n", cudaGetErrorString(cudaError));
            // Handle error appropriately
            exit(0);
        }

        calcDiffBias<<<singleDimblockSize, singleDimnumBlocks10>>>(d_dZ1, d_dZ1_sum, batchSize);
        cudaDeviceSynchronize();
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Kernel launch error (calcDiffBias1): %s\n", cudaGetErrorString(cudaError));
            // Handle error appropriately
            exit(0);
        }

        /// Pointer arith
        d_dZ2 += batchSize * 10;
        d_A2 += batchSize * 10;
        d_one_hot_Y += batchSize * 10;
        d_dZ1 += batchSize * 10;
        d_Z1 += batchSize * 10;
        d_A1 += batchSize * 10;
        d_data += batchSize * 784;
    }

    linearUpdateBias<<<singleDimnumBlocks10, singleDimblockSize>>>(d_dZ2_sum, d_params2, batchSize);
    cudaDeviceSynchronize();
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("Kernel launch error (linearUpdateBias2): %s\n", cudaGetErrorString(cudaError));
        // Handle error appropriately
        exit(0);
    }
    linearUpdateBias<<<singleDimnumBlocks10x10, singleDimblockSize>>>(d_dZ1_sum, d_params1, batchSize);
    cudaDeviceSynchronize();
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("Kernel launch error (linearUpdateBias1): %s\n", cudaGetErrorString(cudaError));
        // Handle error appropriately
        exit(0);
    }
}   

__global__ void one_hot_encode(unsigned char* labels, unsigned char* output, int numLabels, int numClasses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numLabels) {
        // Initialize the output array for this label to zeros
        for (int i = 0; i < numClasses; i++) {
            output[idx * numClasses + i] = 0;
        }

        // Set the element at the index corresponding to the label to 1
        int label = labels[idx];
        if (label < numClasses) {
            output[idx * numClasses + label] = 1;
        }
    }
}

__device__ float getRandomNumber(curandState_t globalState) {
    float random = (2 * curand_uniform(&globalState) )- 1.0f;
    return random;
}

__global__ void initParams(ParametersLinear *params1, ParametersLinear *params2){

    for(int i=0; i<10; i++){
        for(int j=0; j<784; j++){
            curandState_t globalState;
            curand_init(clock64() * i*j, 0, 0, &globalState);
            params1->W[i*784 + j] = getRandomNumber(globalState) * sqrtf(1./784.);
            //printf("W1: %i %f \n", i*784 + j, params1->W[i*784 + j]);
        }
    }
    for(int i=0; i<10; i++){
        curandState_t globalState;
        curand_init(clock64() + i, 0, 0, &globalState);
        params1->B[i] = getRandomNumber(globalState) * sqrtf(1./10.);
        //printf("B1: %i %f \n", i, params1->B[i]);
    }
    for(int i=0; i<10; i++){
        for(int j=0; j<10; j++){
            curandState_t globalState;
            curand_init(clock64()*i,0 , 0, &globalState);
            params2->W[i*10 + j] = getRandomNumber(globalState) * sqrtf(1./20.);
            //printf("W2: %i %f \n", i*10 + j, params2->W[i*10 + j]);
        }
    }
    for(int i=0; i<10; i++){
        curandState_t globalState;
        curand_init(clock64()*i, 0, 0, &globalState);
        params2->B[i] = getRandomNumber(globalState) * sqrtf(1./10);
        //printf("B2: %i %f \n", i, params1->B[i]);
    }
}

__global__ void getPrediction(float *A2, int Z_x_dim, int Z_y_dim, int *predictions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Z_x_dim) {
        float max_val = 0;
        int max_i = 0;
        for (int i = idx * Z_y_dim; i < idx * Z_y_dim + Z_y_dim; i++) {
            if (A2[i] > max_val) {
                max_val = A2[i];
                max_i = i;
                //printf("max_val: %i %i %f \n", i % Z_y_dim, A2[i]);
            }
        }
        predictions[idx] = max_i % Z_y_dim;
    }
    
}

__global__ void _getAccuracy(int *predictions, unsigned char *Y, int numLabels, float *accuracy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numLabels) {
        if(predictions[idx] == Y[idx])
            atomicAdd(accuracy, 1.0);
    }
}

void getAccuracy(float *d_A2, unsigned char *d_labels, int *d_numImages){
    float *d_accuracy;
    int *d_predictions;

    int n_images = 60000;
    cudaError_t err;
    cudaMalloc((void **)&d_accuracy, sizeof(float));
    cudaMalloc((void **)&d_predictions, n_images * sizeof(int));

    int numThreads = 512;
    int numBlocks = (n_images + numThreads - 1) / numThreads;
    cudaError_t cudaError;
    getPrediction<<<numBlocks, numThreads>>>(d_A2, n_images, 10, d_predictions);
    cudaDeviceSynchronize();
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("Kernel launch error (getPrediction): %s\n", cudaGetErrorString(cudaError));
        exit(0);
    }
    _getAccuracy<<<numBlocks, numThreads>>>(d_predictions, d_labels, n_images, d_accuracy);
    cudaDeviceSynchronize();
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("Kernel launch error (_getAccuracy): %s\n", cudaGetErrorString(cudaError));
        // Handle error appropriately
        exit(0);
    }

    float *h_accuracy = (float*)malloc(sizeof(float));
    cudaMemcpy(h_accuracy, d_accuracy, sizeof(float), cudaMemcpyDeviceToHost);
    printf("NAcc: %f \n", *h_accuracy);
    printf("Accuracy: %f \n", *h_accuracy / n_images);
}

void NeuralNetwork(float *h_data, int *h_numImages, int *h_numRows, int *h_numCols, unsigned char *h_labels){

    printf("Init Params \n");
    ParametersLinear* h_params1 = (ParametersLinear*)malloc(sizeof(ParametersLinear));
    ParametersLinear* h_params2 = (ParametersLinear*)malloc(sizeof(ParametersLinear));


    float *d_data;
    int *d_numImages, *d_numRows, *d_numCols;
    ParametersLinear *d_params1, *d_params2;
    cudaMalloc((void**)&d_data, *h_numImages* *h_numRows * *h_numCols * sizeof(float));

    cudaMalloc((void**)&d_numImages, sizeof(int));
    cudaMalloc((void**)&d_numRows, sizeof(int));
    cudaMalloc((void**)&d_numCols, sizeof(int));

    /// Parameters Memory
    cudaMalloc((void**)&d_params1, sizeof(ParametersLinear));
    cudaMalloc((void**)&d_params2, sizeof(ParametersLinear));
  
    float* d_W1;
    cudaMalloc((void**)&d_W1, 10 * 784 * sizeof(float));
    cudaMemcpy(&(d_params1->W), &d_W1, sizeof(float*), cudaMemcpyHostToDevice);

    float* d_B1;
    cudaMalloc((void**)&d_B1, 10 * 1 * sizeof(float));
    cudaMemcpy(&(d_params1->B), &d_B1, sizeof(float*), cudaMemcpyHostToDevice);
    
    
    float* d_W2;
    cudaMalloc((void**)&d_W2, 10 * 10 * sizeof(float));
    cudaMemcpy(&(d_params2->W), &d_W2, sizeof(float*), cudaMemcpyHostToDevice);
    float* d_B2;
    cudaMalloc((void**)&d_B2, 10 * 1 * sizeof(float));
    cudaMemcpy(&(d_params2->B), &d_B2, sizeof(float*), cudaMemcpyHostToDevice);

    cudaError_t cudaError;
    /// Init the device weights

    initParams<<<1, 1>>>(d_params1, d_params2);
    cudaDeviceSynchronize();
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("Kernel launch error (initParams): %s\n", cudaGetErrorString(cudaError));
        // Handle error appropriately
        exit(0);
    }
    
    ///
    
    cudaMemcpy(d_data, h_data, *h_numImages* *h_numRows * *h_numCols * sizeof(float), cudaMemcpyHostToDevice);
    float * d_data_T;
    cudaMalloc((void**)&d_data_T, *h_numImages * 784 * sizeof(float));

    dim3 block_size(16,16);

    int batchSize = 1000;

    for (int i = 0; i < *h_numImages; i += batchSize) {
        int batch_size = min(1000, *h_numImages - i);
        dim3 num_of_blocks((batch_size + block_size.x - 1) / block_size.x, (784 + block_size.y - 1) / block_size.y);
        transposeMatrix<<<num_of_blocks, block_size>>>(d_data + i * 784, d_data_T + i * 784, batch_size, 784);
        cudaDeviceSynchronize();
        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Kernel launch error (transposeMatrix1): %s\n", cudaGetErrorString(cudaError));
            // Handle error appropriately
            exit(0);
        }
    }

    cudaMemcpy(d_numImages, h_numImages, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numRows, h_numRows, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numCols, h_numCols, sizeof(int), cudaMemcpyHostToDevice);
    
    
    /// Init the outputs of the forward steps
    float *d_Z1; // Z1
    cudaMalloc((void**)&d_Z1, *h_numImages * 10 * sizeof(float));
    float *d_A1; // A1
    cudaMalloc((void**)&d_A1, *h_numImages * 10 * sizeof(float));
    float *d_Z2; // Z2
    cudaMalloc((void**)&d_Z2, *h_numImages * 10 * sizeof(float));
    float *d_A2; // A2
    cudaMalloc((void**)&d_A2, *h_numImages * 10 * sizeof(float));

    float *d_dZ1; // Z1
    cudaMalloc((void**)&d_dZ1, *h_numImages * 10 * sizeof(float));
    float *d_dZ2; // Z2
    cudaMalloc((void**)&d_dZ2, *h_numImages * 10 * sizeof(float));

    unsigned char *d_labels;
    unsigned char *d_one_hot;

    int numLabels = 60000;
    int numClasses = 10; 


    cudaMalloc((void**)&d_one_hot, numLabels * numClasses * sizeof(unsigned char));

    cudaMalloc((void**)&d_labels, numLabels * sizeof(unsigned char));
    cudaMemcpy(d_labels, h_labels, numLabels * sizeof(unsigned char), cudaMemcpyHostToDevice);


    int numThreads = 256;
    int numBlocks = (60000 + numThreads - 1) / numThreads;
    
    one_hot_encode<<<numBlocks, numThreads>>>(d_labels, d_one_hot, numLabels, numClasses);
    cudaDeviceSynchronize();
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("Kernel launch error (one_hot_encode): %s\n", cudaGetErrorString(cudaError));
        // Handle error appropriately
        exit(0);
    }
    float *d_A1_orgininal = d_A1;
    float *d_A2_orgininal = d_A2;
    float *d_Z1_orgininal = d_Z1;
    float *d_Z2_orgininal = d_Z2;

    float *d_data_original = d_data;

    int iter = 100;
    for(int i=0; i < iter; i++){
        printf("Iteration: %i \n", i);
        ForwardProp(d_data_T, d_params1, d_params2, d_numRows, d_numCols, h_numImages, h_numRows, h_numCols,
                    d_Z1, d_A1, d_Z2, d_A2);
        
        d_A1 = d_A1_orgininal; 
        d_A2 = d_A2_orgininal; 
        d_Z1 = d_Z1_orgininal; 
        d_Z2 = d_Z2_orgininal;

        d_data = d_data_original;
        
        BackProp(d_Z1, d_A1, d_A2, d_params1, d_params2, d_one_hot, d_data, d_dZ2, d_dZ1, h_numImages, h_numRows, h_numCols);

        d_A1 = d_A1_orgininal; 
        d_A2 = d_A2_orgininal; 
        d_Z1 = d_Z1_orgininal; 
        d_Z2 = d_Z2_orgininal;

        d_data = d_data_original;
        getAccuracy(d_A2, d_labels, d_numImages);
    }
}