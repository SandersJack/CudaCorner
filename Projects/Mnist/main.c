#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "NeuralNetwork.h"

#define MNIST_MAGIC 0x00000803
#define MNIST_NIMAGE 60000
#define MNIST_ROW 28
#define MNIST_Col 28

#define MNIST_LABEL_MAGIC 0x00000801

/**
 * Convert from the big endian format in the dataset if we're on a little endian
 * machine.
 */
uint32_t map_uint32(uint32_t in)
{
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return (
        ((in & 0xFF000000) >> 24) |
        ((in & 0x00FF0000) >>  8) |
        ((in & 0x0000FF00) <<  8) |
        ((in & 0x000000FF) << 24)
    );
#else
    return in;
#endif
}

typedef struct mnist_image_file_header_t_ {
    uint32_t magic_number;
    uint32_t number_of_images;
    uint32_t number_of_rows;
    uint32_t number_of_columns;
} mnist_image_file_header;

typedef struct mnist_label_file_header_t_ {
    uint32_t magic_number;
    uint32_t number_of_labels;
} mnist_label_file_header;

// Function to load MNIST images
float* load_mnist_images(const char* file_path) {
    mnist_image_file_header header;


    FILE *f = fopen(file_path, "rb");

    if (f == NULL) {
        printf("Error opening file %s!\n", file_path);
        exit(1);
    }

    if(fread(&header, sizeof(mnist_image_file_header), 1, f) != 1){
        fprintf(stderr, "Could not read image file header");
        fclose(f);
        return NULL;
    }

    header.magic_number = map_uint32(header.magic_number);
    header.number_of_images = map_uint32(header.number_of_images);
    header.number_of_rows = map_uint32(header.number_of_rows);
    header.number_of_columns = map_uint32(header.number_of_columns);

    if(header.magic_number != MNIST_MAGIC){
        fprintf(stderr, "Magic number of Image File is Incorrect: %i != %i \n", MNIST_MAGIC, header.magic_number);
        exit(1);
    }

    if(header.number_of_images != MNIST_NIMAGE){
        fprintf(stderr, "Number of images of Image File is Incorrect: %i != %i \n", MNIST_NIMAGE, header.number_of_images);
        exit(1);
    }

    if(header.number_of_rows != MNIST_ROW){
        fprintf(stderr, "Number of rows of Image File is Incorrect: %i != %i \n", MNIST_ROW, header.number_of_rows);
        exit(1);
    }

    if(header.number_of_columns != MNIST_Col){
        fprintf(stderr, "Number of cols of Image File is Incorrect: %i != %i \n", MNIST_Col, header.number_of_columns);
        exit(1);
    }

    // Allocate memory for image data
    float* data = (float*)malloc(sizeof(float) * header.number_of_rows * header.number_of_columns * header.number_of_images);
    if (data == NULL) {
        printf("Memory allocation failed!\n");
        exit(1);
    }

    // Read image data
    unsigned char pixel;
    for (int i = 0; i < header.number_of_images; i++) {
        for (int r = 0; r < header.number_of_rows; r++) {
            for (int c = 0; c < header.number_of_columns; c++) {
                fread(&pixel, sizeof(unsigned char), 1, f);
                data[i * header.number_of_columns * header.number_of_columns + r * header.number_of_columns + c] = (float)pixel / 255.0f;
            }
        }
    }


    fclose(f);
    return data;
}

unsigned char* load_mnist_labels(const char* file_path) {
    mnist_label_file_header header;

    FILE *f = fopen(file_path, "rb");

    if (f == NULL) {
        printf("Error opening file %s!\n", file_path);
        exit(1);
    }

    if(fread(&header, sizeof(mnist_label_file_header), 1, f) != 1){
        fprintf(stderr, "Could not read label file header");
        fclose(f);
        return NULL;
    }

    header.magic_number = map_uint32(header.magic_number);
    header.number_of_labels = map_uint32(header.number_of_labels);

    if(header.magic_number != MNIST_LABEL_MAGIC){
        fprintf(stderr, "Magic number of Label File is Incorrect: %i != %i \n", MNIST_LABEL_MAGIC, header.magic_number);
        exit(1);
    }

    if(header.number_of_labels != MNIST_NIMAGE){
        fprintf(stderr, "Number of images of Label File is Incorrect: %i != %i \n", MNIST_NIMAGE, header.number_of_labels);
        exit(1);
    }

    // Allocate memory for label data
    unsigned char* data = (unsigned char*)malloc(sizeof(unsigned char) * header.number_of_labels);
    if (data == NULL) {
        printf("Memory allocation failed!\n");
        exit(1);
    }

    // Read label data
    fread(data, sizeof(unsigned char), header.number_of_labels, f);
    fclose(f);

    return data;
}

void display_mnist_image(const float *image) {
    for (int i = 0; i < MNIST_ROW; ++i) {
        for (int j = 0; j < MNIST_Col; ++j) {
            // Print the pixel value
            if (image[i * MNIST_Col + j] > 0.5) {
                printf("X"); // High pixel value (white)
            } else {
                printf(" "); // Low pixel value (black)
            }
        }
        printf("\n");
    }
}

int main(int argc, char **argv){   
    printf("%s Starting ... \n", argv[0]);

    const char* images_file = "data/train-images.idx3-ubyte";
    const char* labels_file = "data/train-labels.idx1-ubyte";

    printf("Opening Files \n");
    float* images = load_mnist_images(images_file);
    display_mnist_image(images);
    unsigned char* labels = load_mnist_labels(labels_file);
    printf("Label: %i \n", labels[0]);

    int* num_images = (int*)malloc(sizeof(int));
    *num_images = MNIST_NIMAGE;

    int* num_rows = (int*)malloc(sizeof(int));
    *num_rows = MNIST_ROW;

    int* num_cols = (int*)malloc(sizeof(int));
    *num_cols = MNIST_Col;

    NeuralNetwork(images, num_images, num_rows, num_cols, labels);

    free(images);
    free(labels);

    return 0;
}