#ifndef NeuralNetwork_H
#define NeuralNetwork_H

typedef struct 
{
    float *W;
    float *B;
} ParametersLinear;


void NeuralNetwork(float *h_data, int *h_numImages, int *h_numRows, int *h_numCols);

void ForwardProp(float *d_X, ParametersLinear *d_params1, ParametersLinear *d_params2, int *d_numImages, int *d_numRows, 
    int *d_numCols, int *h_numImages, int *h_numRows, int *h_numCols,
    float *d_Z1, float *d_A1, float *d_Z2, float *d_A2);

#endif