#ifndef NeuralNetwork_H
#define NeuralNetwork_H

typedef struct 
{
    float W[10 * 784];
    float B[10 * 1];
} ParametersLinear;


void NeuralNetwork(float *h_data, int *h_numImages, int *h_numRows, int *h_numCols);

void ForwardProp(float *d_A, ParametersLinear *d_params, int *d_numImages, int *d_numRows, int *d_numCols, int *h_numImages, int *h_numRows, int *h_numCols);

#endif