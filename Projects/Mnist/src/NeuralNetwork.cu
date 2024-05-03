#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>

extern "C" {
    #include "NeuralNetwork.h"
}

__global__ void sumExp(float *Z2, float *sum_exp){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int Z_x_dim = 60000;
    int Z_y_dim = 10;

    if(idx < Z_x_dim*Z_y_dim){
        atomicAdd(sum_exp, exp(Z2[idx]));
        //printf("VAL: %i %f \n",idx,  Z2[idx]);
    }
}

__global__ void softMax(float *Z2, float *A2, float *sum_exp, 
    int Z_x_dim, int Z_y_dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < Z_x_dim*Z_y_dim){
        A2[idx] = exp(Z2[idx]) * *sum_exp;
    }
}

__global__ void reLUBack(float* Z1, float* dA1, int Z_x_dim, int Z_y_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < Z_x_dim * Z_y_dim) {
        if(Z1[idx] > 0){
            dA1[idx] *= Z1[idx];
        } else {
            dA1[idx] *= 0;
        }
    }
}

__global__ void reLUForward(float *Z1, float *A1, int Z_x_dim, int Z_y_dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < Z_x_dim * Z_y_dim){
        if(Z1[idx] > 0){
            //A1[idx] = Z1[idx];
        } else {
            A1[0] = 0;
        }
    }
}

__global__ void linearForwardProp(float* A, float* Z, ParametersLinear *params, int *num_images, int *num_rows, int *num_cols,
    int Z_x_dim, int Z_y_dim, int W_x_dim, int W_y_dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    float Z_value = 0;
    
    if(idx < Z_x_dim && idy < Z_y_dim){
        for(int t=0; t< W_y_dim; t++){
            Z_value += A[idx*W_y_dim + t] * params->W[idy * W_y_dim + t] + params->B[idy];
        }
        Z[idx * W_x_dim + idy] = Z_value;
    }
}

__global__ void linearBackProp(float *dZ2, float *dZ1, ParametersLinear *params, int Z_x_dim, int Z_y_dim, int W_y_dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    float dZ1_value = 0.0f;

    if(idx < Z_x_dim && idy < Z_y_dim){
        for(int i=0; i<W_y_dim; i++){
            dZ1_value += params->W[i*W_y_dim + idx] * dZ2[i*W_y_dim + idy];
        }
        dZ2[idx*Z_y_dim+idy] = dZ1_value;
    }
}

__global__ void linearUpdateWeight(float* A, float* dZ, ParametersLinear *params, int W_x_dim, int W_y_dim, int A_x_dim, int A_y_dim, int dZ_x_dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    float dW_value = 0.0f;

    float learning_rate = 0.01;

    if(idx < W_x_dim && idy < W_y_dim){
        for(int i=0; i<dZ_x_dim; i++){
            dW_value += dZ[idx*W_y_dim + i] * A[idy * A_y_dim + i];
        }

        params->W[idx * W_y_dim + idy] = params->W[idx * W_y_dim + idy] - learning_rate * dW_value/A_x_dim;
    }
}

__global__ void linearUpdateBias(float *dZ, ParametersLinear *params, int dZ_x_dim, int dZ_y_dim){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float learning_rate = 0.01;

    if(idx < dZ_x_dim * dZ_y_dim){
        int dZ_x = idx / dZ_x_dim;
        int dZ_y = idx % dZ_x_dim;
        atomicAdd(&params->B[dZ_y], -learning_rate * (dZ[dZ_x * dZ_y_dim + dZ_y] / dZ_y_dim));
    }
}

void ForwardProp(float *d_X, ParametersLinear *d_params1, ParametersLinear *d_params2, int *d_numImages, int *d_numRows, 
    int *d_numCols, int *h_numImages, int *h_numRows, int *h_numCols,
    float *d_Z1, float *d_A1, float *d_Z2, float *d_A2){

    float *d_sum_exp; 
    cudaMalloc((void**)&d_sum_exp, sizeof(float));

    int matrixSize = *h_numImages * 784;
    printf("Start Forward \n");
    cudaError_t cudaError;

    int batchSize = 1000;

    int numBatches = (*h_numImages) / batchSize;

    int batcharraysize = batchSize * 784;

    dim3 block_size(16,16);
    dim3 num_of_blocks((batchSize+block_size.x-1)/block_size.x, (*h_numRows * *h_numCols+block_size.y-1)/block_size.y);

    dim3 num_of_blocks2((batchSize+block_size.x-1)/block_size.x, (10+block_size.y-1)/block_size.y);

    int singleDimblockSize = 1028;
    int singleDimnumBlocks = (batcharraysize + singleDimblockSize - 1) / singleDimblockSize;

    int singleDimnumBlocks2 = (batchSize * 10 + singleDimblockSize - 1) / singleDimblockSize;

    printf("Matrix Size %i \n", matrixSize);
    printf("Num block x threads %i \n", singleDimblockSize * singleDimnumBlocks);
    printf("Num Threads %i \n", singleDimnumBlocks);

    for (int i = 0; i < numBatches; i++) {
        printf("Batch %i Start \n", i);
        // Calculate the start and end indices for this batch
        int startIdx = i * batchSize;
        int endIdx = min(startIdx + batchSize, *h_numImages);
        // 47040000
        // 10976000
        // Calculate the size of this batch
        int batchMatrixSize = (endIdx - startIdx) * 784;
        printf("%i \n", startIdx * 784);
        /// First Linear Layer
        linearForwardProp<<<num_of_blocks, block_size>>>(d_X, d_Z1, d_params1, d_numImages, d_numRows, d_numCols,
            batchSize, 10, 10 , 784);
        cudaDeviceSynchronize();
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Kernel launch error (linearForwardProp1): %s\n", cudaGetErrorString(cudaError));
            // Handle error appropriately
            exit(0);
        }
        /// Relu Layer
        reLUForward<<<singleDimblockSize, singleDimnumBlocks>>>(d_Z1, d_A1, batchSize, 10);
        cudaDeviceSynchronize();

        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Kernel launch error (reLUForward): %s\n", cudaGetErrorString(cudaError));
            // Handle error appropriately
            exit(0);
        }
        /// Second Linear Layer
        linearForwardProp<<<num_of_blocks2, block_size>>>(d_A1, d_Z2, d_params2, d_numImages, d_numRows, d_numCols, 
            batchSize, 10, 10, 10);
        cudaDeviceSynchronize();
        
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Kernel launch error (linearForwardProp2): %s\n", cudaGetErrorString(cudaError));
            // Handle error appropriately
            exit(0);
        }
        
        /// Sum the exponetial 
        sumExp<<<singleDimblockSize, singleDimnumBlocks2>>>(d_Z2, d_sum_exp);
        cudaDeviceSynchronize();
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Kernel launch error (sumExp): %s\n", cudaGetErrorString(cudaError));
            // Handle error appropriately
            exit(0);
        }
        /// Conduct the SoftMax
        softMax<<<singleDimblockSize, singleDimnumBlocks2>>>(d_Z2, d_A2, d_sum_exp, 
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

void BackProp(float *d_Z1, float *d_A1, float *d_A2, float *d_W2, ParametersLinear *d_params1, ParametersLinear *d_params2, unsigned char *d_one_hot_Y, 
    float *d_data, float* d_dZ2, float *d_dZ1, int *h_numImages, int *h_numRows, int *h_numCols){
    
    printf("Start Back \n");
    cudaError_t cudaError;

    int batchSize = 1000;

    int numBatches = (*h_numImages) / batchSize;

    int batcharraysize = batchSize * 784;
    int batcharraysize10 = batchSize * 10;

    dim3 block_size(8,8);
    dim3 num_of_blocks((batchSize+block_size.x-1)/block_size.x,(*h_numRows * *h_numCols+block_size.y-1)/block_size.y);

    int singleDimblockSize = 1028;
    int singleDimnumBlocks784 = (batcharraysize + singleDimblockSize - 1) / singleDimblockSize;
    int singleDimnumBlocks10 = (batcharraysize10 + singleDimblockSize - 1) / singleDimblockSize;

    for (int i = 0; i < numBatches; i++) {
        printf("Batch %i Start \n", i);
        // Calculate the start and end indices for this batch
        int startIdx = i * batchSize;
        int endIdx = min(startIdx + batchSize, *h_numImages);

        // Calculate the size of this batch
        int batchMatrixSize = (endIdx - startIdx) * 784;
        printf("%i \n", startIdx * 10);

        startBackProp<<<singleDimblockSize, singleDimnumBlocks10>>>(d_dZ2, d_A2, d_one_hot_Y, 60000, 10);
        cudaDeviceSynchronize();
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Kernel launch error (startBackProp): %s\n", cudaGetErrorString(cudaError));
            // Handle error appropriately
            exit(0);
        }

        linearBackProp<<<num_of_blocks, block_size>>>(d_dZ2, d_dZ1, d_params2, 60000, 10, 10);
        cudaDeviceSynchronize();
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Kernel launch error (linearBackProp): %s\n", cudaGetErrorString(cudaError));
            // Handle error appropriately
            exit(0);
        }
        
        reLUBack<<<singleDimblockSize, singleDimnumBlocks10>>>(d_Z1, d_dZ1, 60000, 10);
        cudaDeviceSynchronize();
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Kernel launch error (reLUBack): %s\n", cudaGetErrorString(cudaError));
            // Handle error appropriately
            exit(0);
        }
        // Linear 2
        linearUpdateWeight<<<num_of_blocks, singleDimnumBlocks10>>>(d_A1, d_dZ2, d_params2, 10, 10, 60000, 10, 60000);
        cudaDeviceSynchronize();
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Kernel launch error (linearUpdateWeight2): %s\n", cudaGetErrorString(cudaError));
            // Handle error appropriately
            exit(0);
        }
        linearUpdateBias<<<singleDimblockSize, singleDimnumBlocks10>>>(d_dZ2, d_params2, 60000, 10);
        cudaDeviceSynchronize();
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Kernel launch error (linearUpdateWeight2): %s\n", cudaGetErrorString(cudaError));
            // Handle error appropriately
            exit(0);
        }
        // Linear 1        
        linearUpdateWeight<<<num_of_blocks, block_size>>>(d_data, d_dZ1, d_params1, 784, 10, 784, 10, 60000);
        cudaDeviceSynchronize();
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Kernel launch error (linearUpdateWeight2): %s\n", cudaGetErrorString(cudaError));
            // Handle error appropriately
            exit(0);
        }
        linearUpdateBias<<<singleDimblockSize, singleDimnumBlocks10>>>(d_dZ1, d_params2, 60000, 10);
        cudaDeviceSynchronize();
        cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            printf("Kernel launch error (linearUpdateWeight2): %s\n", cudaGetErrorString(cudaError));
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
}   

__global__ void one_hot_encode(unsigned char* labels, unsigned char* output, int numLabels, int numClasses) {
    int idx = threadIdx.x * blockIdx.x * blockDim.x;

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
    float random = curand_uniform(&globalState) - 0.5f;
    return random;
}

__global__ void initParams(ParametersLinear *params1, ParametersLinear *params2){
    

    for(int i=0; i<10; i++){
        for(int j=0; j<784; j++){
            curandState_t globalState;
            curand_init(clock64() * i*j, 0, 0, &globalState);
            params1->W[i*784 + j] = getRandomNumber(globalState) / sqrtf(1./784.);
            //printf("VAL: %i %f \n", i*784 + j, params1->W[i*784 + j]);
        }
    }
    for(int i=0; i<10; i++){
        curandState_t globalState;
        curand_init(clock64() + i, 0, 0, &globalState);
        params1->B[i] = getRandomNumber(globalState) / sqrtf(1./10.);
    }
    for(int i=0; i<10; i++){
        for(int j=0; j<10; j++){
            curandState_t globalState;
            curand_init(clock64(),0 , 0, &globalState);
            params2->W[i*10 + j] = getRandomNumber(globalState) / sqrtf(1./20.);
        }
    }
    for(int i=0; i<10; i++){
        curandState_t globalState;
        curand_init(clock64()*i, 0, 0, &globalState);
        params2->B[i] = getRandomNumber(globalState) / sqrtf(1./10);
    }
}

__global__ void getPrediction(float *A2, int Z_x_dim, int Z_y_dim, int *predictions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Z_x_dim) {
        float max_val = A2[idx];
        int max_idx = idx;
        for (int i = idx * Z_y_dim; i < Z_x_dim * Z_y_dim; i++) {
            if (A2[i] > max_val) {
                max_val = A2[i];
                max_idx = idx;
            }
        }
        predictions[max_idx / 10] = max_idx % Z_y_dim;
    }
}

__global__ void _getAccuracy(int *predictions, unsigned char *Y, int numLabels, float *accuracy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numLabels) {
        *accuracy += (predictions[idx] == Y[idx]) ? 1.0f : 0.0f;
    }
}

void getAccuracy(float *d_A2, unsigned char *d_labels, int *d_numImages){
    float *d_accuracy;
    int *d_predictions;
    cudaError_t err;
    printf("Start Acc \n");
    cudaMalloc((void **)&d_accuracy, sizeof(float));
    cudaMalloc((void **)&d_predictions, 60000 * sizeof(int));

    printf("Start 1 \n");
    int numThreads = 512;
    int numBlocks = (10 + numThreads - 1) / numThreads;
    cudaError_t cudaError;
    printf("Start 2 \n");
    getPrediction<<<numBlocks, numThreads>>>(d_A2, *d_numImages, 10, d_predictions);
    cudaDeviceSynchronize();
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("Kernel launch error (getPrediction): %s\n", cudaGetErrorString(cudaError));
        exit(0);
    }
    printf("Start 3 \n");
    _getAccuracy<<<numBlocks, numThreads>>>(d_predictions, d_labels, *d_numImages, d_accuracy);
    cudaDeviceSynchronize();
    cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        printf("Kernel launch error (_getAccuracy): %s\n", cudaGetErrorString(cudaError));
        // Handle error appropriately
        exit(0);
    }

    float *h_accuracy = (float*)malloc(sizeof(float));
    cudaMemcpy(h_accuracy, d_accuracy, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Accuracy: %f \n", *h_accuracy / 60000);
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


    int numThreads = 512;
    int numBlocks = (numLabels + numThreads - 1) / numThreads;
    
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
    // Testing with one forward prop
    ForwardProp(d_data, d_params1, d_params2, d_numImages, d_numRows, d_numCols, h_numImages, h_numRows, h_numCols,
                d_Z1, d_A1, d_Z2, d_A2);

    d_A1 = d_A1_orgininal; 
    d_A2 = d_A2_orgininal; 
    d_Z1 = d_Z1_orgininal; 
    d_Z2 = d_Z2_orgininal;

    d_data = d_data_original;
    
    BackProp(d_Z1, d_A1, d_A2, d_W2, d_params1, d_params2, d_one_hot, d_data, d_dZ2, d_dZ1, h_numImages, h_numRows, h_numCols);

    d_A1 = d_A1_orgininal; 
    d_A2 = d_A2_orgininal; 
    d_Z1 = d_Z1_orgininal; 
    d_Z2 = d_Z2_orgininal;

    d_data = d_data_original;
    getAccuracy(d_A2, d_labels, d_numImages);
}