#ifndef NeuralNetwork_H
#define NeuralNetwork_H

typedef struct 
{
    float W1[10 * 784];
    float B1[10 * 1];
    float W2[10 * 10];
    float B2[10 * 1];
} Parameters;


void NeuralNetwork(float *h_data, int *h_numImages, int *h_numRows, int *h_numCols);

void ForwardProp(float *d_A, Parameters *d_params, int *d_numImages, int *d_numRows, int *d_numCols, int *h_numImages, int *h_numRows, int *h_numCols);

#endif