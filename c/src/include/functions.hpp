#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

#include <iostream>
#include "ac_fixed.h"
#include "ac_math/ac_relu.h"

// defining values
#define IMG_HEIGHT 24
#define IMG_WIDTH 24
#define IMG_CHANNELS 3

#define CONV1_FILTER_HEIGHT 3
#define CONV1_FILTER_WIDTH 3
#define CONV1_FILTER_CHANNELS 3
#define CONV1_FILTER_NUMBER 64
#define CONV1_BIAS_NUMBER 64

#define MAXPOOL1_IN_HEIGHT 24
#define MAXPOOL1_IN_WIDTH 24
#define MAXPOOL1_IN_CHANNELS 64

#define MAXPOOL1_OUT_HEIGHT 12
#define MAXPOOL1_OUT_WIDTH 12
#define MAXPOOL1_OUT_CHANNELS 64


#define CONV2_FILTER_HEIGHT 3
#define CONV2_FILTER_WIDTH 3
#define CONV2_FILTER_CHANNELS 64
#define CONV2_FILTER_NUMBER 32
#define CONV2_BIAS_NUMBER 32

#define MAXPOOL2_IN_HEIGHT 12
#define MAXPOOL2_IN_WIDTH 12
#define MAXPOOL2_IN_CHANNELS 32

#define MAXPOOL2_OUT_HEIGHT 6
#define MAXPOOL2_OUT_WIDTH 6
#define MAXPOOL2_OUT_CHANNELS 32


#define CONV3_FILTER_HEIGHT 3
#define CONV3_FILTER_WIDTH 3
#define CONV3_FILTER_CHANNELS 32
#define CONV3_FILTER_NUMBER 20
#define CONV3_BIAS_NUMBER 20

#define MAXPOOL3_IN_HEIGHT 6
#define MAXPOOL3_IN_WIDTH 6
#define MAXPOOL3_IN_CHANNELS 20

#define MAXPOOL3_OUT_HEIGHT 3
#define MAXPOOL3_OUT_WIDTH 3
#define MAXPOOL3_OUT_CHANNELS 20


#define MAXPOOL_FILTER_WIDTH 3
#define MAXPOOL_FILTER_DEPTH 3

fixed_t ReLu(fixed_t x);
void softmax(fixed_t x[10]);
void reshape(fixed_t input[3][3][20], fixed_t output[180]);
void FCP(fixed_t M[180], fixed_t weights[180][10], fixed_t bias[10], fixed_t output[10]);

void convolution1(
    fixed_t image[IMG_HEIGHT][IMG_WIDTH][IMG_CHANNELS],
    fixed_t Ks[CONV1_FILTER_HEIGHT][CONV1_FILTER_WIDTH][CONV1_FILTER_CHANNELS][CONV1_FILTER_NUMBER],
    fixed_t biais[CONV1_BIAS_NUMBER],
    fixed_t output[MAXPOOL1_IN_HEIGHT][MAXPOOL1_IN_WIDTH][MAXPOOL1_IN_CHANNELS]
);

void convolution2(
    fixed_t image[MAXPOOL1_OUT_HEIGHT][MAXPOOL1_OUT_WIDTH][MAXPOOL1_OUT_CHANNELS],
    fixed_t Ks[CONV2_FILTER_HEIGHT][CONV2_FILTER_WIDTH][CONV2_FILTER_CHANNELS][CONV2_FILTER_NUMBER],
    fixed_t biais[CONV2_BIAS_NUMBER],
    fixed_t output[MAXPOOL2_IN_HEIGHT][MAXPOOL2_IN_WIDTH][MAXPOOL2_IN_CHANNELS]
);

void convolution3(
    fixed_t image[MAXPOOL2_OUT_HEIGHT][MAXPOOL2_OUT_WIDTH][MAXPOOL2_OUT_CHANNELS],
    fixed_t Ks[CONV3_FILTER_HEIGHT][CONV3_FILTER_WIDTH][CONV3_FILTER_CHANNELS][CONV3_FILTER_NUMBER],
    fixed_t biais[CONV3_BIAS_NUMBER],
    fixed_t output[MAXPOOL3_IN_HEIGHT][MAXPOOL3_IN_WIDTH][MAXPOOL3_IN_CHANNELS]
);

void maxpool1(
    fixed_t input[MAXPOOL1_IN_HEIGHT][MAXPOOL1_IN_WIDTH][MAXPOOL1_IN_CHANNELS],
    fixed_t output[MAXPOOL1_OUT_HEIGHT][MAXPOOL1_OUT_WIDTH][MAXPOOL1_OUT_CHANNELS]
);

void maxpool2(
    fixed_t input[MAXPOOL2_IN_HEIGHT][MAXPOOL2_IN_WIDTH][MAXPOOL2_IN_CHANNELS],
    fixed_t output[MAXPOOL2_OUT_HEIGHT][MAXPOOL2_OUT_WIDTH][MAXPOOL2_OUT_CHANNELS]
);

void maxpool3(
    fixed_t input[MAXPOOL3_IN_HEIGHT][MAXPOOL3_IN_WIDTH][MAXPOOL3_IN_CHANNELS],
    fixed_t output[MAXPOOL3_OUT_HEIGHT][MAXPOOL3_OUT_WIDTH][MAXPOOL3_OUT_CHANNELS]
);


// float ReLu(float x);
// void softmax(float x[10]);
// void reshape(float input[3][3][20], float output[180]);
// void FCP(float M[180], float weights[180][10], float bias[10], float output[10]);



template <int HEIGHT, int WIDTH, int NUM_CHANNELS, int OUT_HEIGHT, int OUT_WIDTH>
void maxpool_3x3(float input[HEIGHT][WIDTH][NUM_CHANNELS],
                 float output[OUT_HEIGHT][OUT_WIDTH][NUM_CHANNELS]) {
    int c = 0;
    int h = 0;
    int w = 0;
    int start_h, start_w;
    for (c = 0; c < NUM_CHANNELS; c++) {
        for (h = 0; h < OUT_HEIGHT; h++) {
            for (w = 0; w < OUT_WIDTH; w++) {
                start_h = h * 2;
                start_w = w * 2;
                float max_val = -1; // <-------- plus petit possible
                
                if (start_h + 3 > HEIGHT) {
                    if (start_w + 3 > WIDTH) {
                        // region = M[start_h:y, start_w:x, c]
                        for (int i = start_h; i < HEIGHT; i++) {
                            for (int j = start_w; j < WIDTH; j++) {
                                max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
                            }
                        }
                    }
                    else {
                        // region = M[start_h:y, start_w:start_w+3, c]
                        for (int i = start_h; i < HEIGHT; i++) {
                            for (int j = start_w; j < start_w + 3; j++) {
                                max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
                            }
                        }
                    }
                }
                else if (start_w + 3 > WIDTH) {
                    // region = M[start_h:start_h+3, start_w:x, c]
                    for (int i = start_h; i < start_h + 3; i++) {
                        for (int j = start_w; j < WIDTH; j++) {
                            max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
                        }
                    }
                }
                else {
                    // region = M[start_h:start_h+3, start_w:start_w+3, c]
                    for (int i = start_h; i < start_h + 3; i++) {
                        for (int j = start_w; j < start_w + 3; j++) {
                            max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
                        }
                    }
                }
                output[h][w][c] = max_val;
            }
        }
    }
}

template <int IMG_H, int IMG_W, int IMG_C, int K_H, int K_W, int NUM_FILTERS>
void convolution(const float image[IMG_H][IMG_W][IMG_C],
                 const float Ks[K_H][K_W][IMG_C][NUM_FILTERS],
                 const float biais[NUM_FILTERS],
                 float output[IMG_H][IMG_W][NUM_FILTERS]) {

    int PADDING = 1;
    int PADDED_H = IMG_H + 2 * PADDING;
    int PADDED_W = IMG_W + 2 * PADDING;

    float padded_image[PADDED_H][PADDED_W][IMG_C];
    // padding
    for (int c = 0; c < IMG_C; ++c) {
        for (int i = 0; i < IMG_H + 2 * PADDING; ++i) {
            for (int j = 0; j < IMG_W + 2 * PADDING; ++j) {
                if (i == 0 || j == 0 || i == IMG_H + PADDING || j == IMG_W + PADDING) {
                    padded_image[i][j][c] = 0.0;
                }
                else {
                    padded_image[i][j][c] = image[i - 1][j - 1][c];
                }
            }
        }
    }

    // std::cout << "padded_image:\n";
    // for (int i = 0; i < 5; ++i) {
    //     for (int j = 0; j < 5; ++j) {
    //         std::cout << padded_image[i][j][0] << " ";
    //     }
    //     std::cout << "\n";
    // }

    // init biais
    for (int f = 0; f < NUM_FILTERS; ++f) {
        for (int i = 0; i < IMG_H; ++i) {
            for (int j = 0; j < IMG_W; ++j) {
                output[i][j][f] = biais[f];
            }
        }
    }

    // etape conv
    for (int f = 0; f < NUM_FILTERS; ++f) {
        for (int c = 0; c < IMG_C; ++c) {
            for (int i = 0; i < IMG_H; ++i) {
                for (int j = 0; j < IMG_W; ++j) {
                    float conv_sum = 0;
                    for (int ki = 0; ki < K_H; ++ki) {
                        for (int kj = 0; kj < K_W; ++kj) {
                            conv_sum = conv_sum + padded_image[i + ki][j + kj][c] * Ks[ki][kj][c][f];
                        }
                    }
                    output[i][j][f] = output[i][j][f] + conv_sum;
                }
            }
        }

        // etape relu
        for (int i = 0; i < IMG_H; ++i) {
            for (int j = 0; j < IMG_W; ++j) {
                output[i][j][f] = ReLu(output[i][j][f]);
            }
        }
    }
}

// template <int IMG_H, int IMG_W, int IMG_C>
// void normalize_image(float image[IMG_H][IMG_W][IMG_C],
//                      float normalized[IMG_H][IMG_W][IMG_C]) {
//     const int N = IMG_H * IMG_W * IMG_C;

//     float total = 0.0f;
//     for (int i = 0; i < IMG_H; ++i) {
//         for (int j = 0; j < IMG_W; ++j) {
//             for (int k = 0; k < IMG_C; ++k) {
//                 total += image[i][j][k];
//             }
//         }
//     }
//     float mean_value = total / N;

//     float variance = 0.0f;
//     for (int i = 0; i < IMG_H; ++i) {
//         for (int j = 0; j < IMG_W; ++j) {
//             for (int k = 0; k < IMG_C; ++k) {
//                 float diff = image[i][j][k] - mean_value;
//                 variance += diff * diff;
//             }
//         }
//     }
//     float sigma = std::sqrt(variance / N);

//     std::cout << "Mean: " << mean_value << " ------- Sigma: " << sigma << std::endl;

//     const float epsilon = 1.0f / std::sqrt(N); // Small value to avoid division by zero
//     float denominator = std::max(sigma, epsilon);

//     for (int i = 0; i < IMG_H; ++i) {
//         for (int j = 0; j < IMG_W; ++j) {
//             for (int k = 0; k < IMG_C; ++k) {
//                 normalized[i][j][k] = (image[i][j][k] - mean_value) / denominator;
//             }
//         }
//     }
// }


#endif