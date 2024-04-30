#ifndef NeuralNetwork_H
#define NeuralNetwork_H

typedef struct 
{
    float *W;
    float *B;
} ParametersLinear;


void NeuralNetwork(float *h_data, int *h_numImages, int *h_numRows, int *h_numCols, unsigned char *h_labels);

void ForwardProp(float *d_X, ParametersLinear *d_params1, ParametersLinear *d_params2, int *d_numImages, int *d_numRows, 
    int *d_numCols, int *h_numImages, int *h_numRows, int *h_numCols,
    float *d_Z1, float *d_A1, float *d_Z2, float *d_A2);

void BackProp(float *d_Z1, float *d_A1, float *d_A2, float *d_W2, ParametersLinear *d_params2, unsigned char *d_one_hot_Y, 
    float *d_data, float* d_dZ2, float *d_dZ1);

#endif