#include "functions.hpp"
#include <math.h>

#include <iostream>
#include <cstdint>
// #include "ac_fixed.h"
// #include "ac_math/ac_relu.h"

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


// fixed_t ReLu(fixed_t x) {
//     fixed_t result;
//     ac_math::ac_relu_pwl(x, result); // Version optimisée de ReLU
//     return result;
// }

// void softmax(fixed_t x[10]) {
//     fixed_t sum = 0.0;
//     int i = 0;

//     for (i = 0; i < 10; i++) {
//         x[i] = expf(x[i]);
//         sum += x[i];
//     }

//     for (int i = 0; i < 10; i++) {
//         x[i] /= sum;
//     }
// }

// void reshape(fixed_t input[3][3][20], fixed_t output[180]) {
//     int index = 0;
//     for (int i = 0; i < 3; i++) {
//         for (int j = 0; j < 3; j++) {
//             for (int k = 0; k < 20; k++) {
//                 output[index++] = input[i][j][k];
//             }
//         }
//     }
// }

// void FCP(fixed_t M[180], fixed_t weights[180][10], fixed_t bias[10], fixed_t output[10]) {
//     // NON  TESTE ATTENTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//     fixed_t temp_output[10] = {0};

//     for (int i = 0; i < 10; i++) {
//         for (int j = 0; j < 180; j++) {
//             temp_output[i] += M[j] * weights[j][i];
//         }
//         temp_output[i] += bias[i];
//     }

//     softmax(temp_output);

//     for (int i = 0; i < 10; i++) {
//         output[i] = temp_output[i];
//     }
// }

// void convolution1(
//     fixed_t image[IMG_HEIGHT][IMG_WIDTH][IMG_CHANNELS],
//     fixed_t Ks[CONV1_FILTER_HEIGHT][CONV1_FILTER_WIDTH][CONV1_FILTER_CHANNELS][CONV1_FILTER_NUMBER],
//     fixed_t biais[CONV1_BIAS_NUMBER],
//     fixed_t output[MAXPOOL1_IN_HEIGHT][MAXPOOL1_IN_WIDTH][MAXPOOL1_IN_CHANNELS]
// ) {
//     const int PADDING = 1;
//     const int PADDED_H = IMG_HEIGHT + 2 * PADDING;
//     const int PADDED_W = IMG_WIDTH + 2 * PADDING;

//     fixed_t padded_image[PADDED_H][PADDED_W][IMG_CHANNELS];

//     // Étape 1 : Padding
//     ly_pad: for (int c = 0; c < IMG_CHANNELS; ++c) {
//         lx_pad: for (int i = 0; i < PADDED_H; ++i) {
//             lz_pad: for (int j = 0; j < PADDED_W; ++j) {
//                 if (i < PADDING || j < PADDING || i >= PADDED_H - PADDING || j >= PADDED_W - PADDING) {
//                     padded_image[i][j][c] = 0.0;
//                 } else {
//                     padded_image[i][j][c] = image[i - PADDING][j - PADDING][c];
//                 }
//             }
//         }
//     }

//     // Étape 2 : Initialisation des sorties avec les biais
//     ly_init: for (int f = 0; f < CONV1_FILTER_NUMBER; ++f) {
//         lx_init: for (int i = 0; i < IMG_HEIGHT; ++i) {
//             lz_init: for (int j = 0; j < IMG_WIDTH; ++j) {
//                 output[i][j][f] = biais[f];
//             }
//         }
//     }

//     // Étape 3 : Convolution
//     ly_conv: for (int f = 0; f < CONV1_FILTER_NUMBER; ++f) {
//         lx_conv: for (int c = 0; c < IMG_CHANNELS; ++c) {
//             lz_conv: for (int i = 0; i < IMG_HEIGHT; ++i) {
//                 lw_conv: for (int j = 0; j < IMG_WIDTH; ++j) {
//                     fixed_t conv_sum = 0;
//                     ly_k: for (int ki = 0; ki < CONV1_FILTER_HEIGHT; ++ki) {
//                         lx_k: for (int kj = 0; kj < CONV1_FILTER_WIDTH; ++kj) {
//                             conv_sum += padded_image[i + ki][j + kj][c] * Ks[ki][kj][c][f];
//                         }
//                     }
//                     output[i][j][f] += conv_sum;
//                 }
//             }
//         }
//     }

//     // Étape 4 : Activation ReLU
//     ly_relu: for (int f = 0; f < CONV1_FILTER_NUMBER; ++f) {
//         lx_relu: for (int i = 0; i < IMG_HEIGHT; ++i) {
//             lz_relu: for (int j = 0; j < IMG_WIDTH; ++j) {
//                 output[i][j][f] = ReLu(output[i][j][f]);
//             }
//         }
//     }
// }

// void convolution2(
//     fixed_t image[MAXPOOL1_OUT_HEIGHT][MAXPOOL1_OUT_WIDTH][MAXPOOL1_OUT_CHANNELS],
//     fixed_t Ks[CONV2_FILTER_HEIGHT][CONV2_FILTER_WIDTH][CONV2_FILTER_CHANNELS][CONV2_FILTER_NUMBER],
//     fixed_t biais[CONV2_BIAS_NUMBER],
//     fixed_t output[MAXPOOL2_IN_HEIGHT][MAXPOOL2_IN_WIDTH][MAXPOOL2_IN_CHANNELS]
// ) {
//     const int PADDING = 1;
//     const int PADDED_H = MAXPOOL1_OUT_HEIGHT + 2 * PADDING;
//     const int PADDED_W = MAXPOOL1_OUT_WIDTH + 2 * PADDING;

//     fixed_t padded_image[PADDED_H][PADDED_W][IMG_CHANNELS];

//     // Étape 1 : Padding
//     ly_pad: for (int c = 0; c < MAXPOOL1_OUT_CHANNELS; ++c) {
//         lx_pad: for (int i = 0; i < PADDED_H; ++i) {
//             lz_pad: for (int j = 0; j < PADDED_W; ++j) {
//                 if (i < PADDING || j < PADDING || i >= PADDED_H - PADDING || j >= PADDED_W - PADDING) {
//                     padded_image[i][j][c] = 0.0;
//                 } else {
//                     padded_image[i][j][c] = image[i - PADDING][j - PADDING][c];
//                 }
//             }
//         }
//     }

//     // Étape 2 : Initialisation des sorties avec les biais
//     ly_init: for (int f = 0; f < CONV2_FILTER_NUMBER; ++f) {
//         lx_init: for (int i = 0; i < MAXPOOL1_OUT_HEIGHT; ++i) {
//             lz_init: for (int j = 0; j < MAXPOOL1_OUT_WIDTH; ++j) {
//                 output[i][j][f] = biais[f];
//             }
//         }
//     }

//     // Étape 3 : Convolution
//     ly_conv: for (int f = 0; f < CONV2_FILTER_NUMBER; ++f) {
//         lx_conv: for (int c = 0; c < IMG_CHANNELS; ++c) {
//             lz_conv: for (int i = 0; i < MAXPOOL1_OUT_HEIGHT; ++i) {
//                 lw_conv: for (int j = 0; j < MAXPOOL1_OUT_WIDTH; ++j) {
//                     fixed_t conv_sum = 0;
//                     ly_k: for (int ki = 0; ki < CONV2_FILTER_HEIGHT; ++ki) {
//                         lx_k: for (int kj = 0; kj < CONV2_FILTER_WIDTH; ++kj) {
//                             conv_sum += padded_image[i + ki][j + kj][c] * Ks[ki][kj][c][f];
//                         }
//                     }
//                     output[i][j][f] += conv_sum;
//                 }
//             }
//         }
//     }

//     // Étape 4 : Activation ReLU
//     ly_relu: for (int f = 0; f < CONV2_FILTER_NUMBER; ++f) {
//         lx_relu: for (int i = 0; i < MAXPOOL1_OUT_HEIGHT; ++i) {
//             lz_relu: for (int j = 0; j < MAXPOOL1_OUT_WIDTH; ++j) {
//                 output[i][j][f] = ReLu(output[i][j][f]);
//             }
//         }
//     }
// }

// void convolution3(
//     fixed_t image[MAXPOOL2_OUT_HEIGHT][MAXPOOL2_OUT_WIDTH][MAXPOOL2_OUT_CHANNELS],
//     fixed_t Ks[CONV3_FILTER_HEIGHT][CONV3_FILTER_WIDTH][CONV3_FILTER_CHANNELS][CONV3_FILTER_NUMBER],
//     fixed_t biais[CONV3_BIAS_NUMBER],
//     fixed_t output[MAXPOOL3_IN_HEIGHT][MAXPOOL3_IN_WIDTH][MAXPOOL3_IN_CHANNELS]
// ) {
//     const int PADDING = 1;
//     const int PADDED_H = MAXPOOL2_OUT_HEIGHT + 2 * PADDING;
//     const int PADDED_W = MAXPOOL2_OUT_WIDTH + 2 * PADDING;

//     fixed_t padded_image[PADDED_H][PADDED_W][IMG_CHANNELS];

//     // Étape 1 : Padding
//     ly_pad: for (int c = 0; c < MAXPOOL2_OUT_CHANNELS; ++c) {
//         lx_pad: for (int i = 0; i < PADDED_H; ++i) {
//             lz_pad: for (int j = 0; j < PADDED_W; ++j) {
//                 if (i < PADDING || j < PADDING || i >= PADDED_H - PADDING || j >= PADDED_W - PADDING) {
//                     padded_image[i][j][c] = 0.0;
//                 } else {
//                     padded_image[i][j][c] = image[i - PADDING][j - PADDING][c];
//                 }
//             }
//         }
//     }

//     // Étape 2 : Initialisation des sorties avec les biais
//     ly_init: for (int f = 0; f < CONV3_FILTER_NUMBER; ++f) {
//         lx_init: for (int i = 0; i < MAXPOOL2_OUT_HEIGHT; ++i) {
//             lz_init: for (int j = 0; j < MAXPOOL2_OUT_WIDTH; ++j) {
//                 output[i][j][f] = biais[f];
//             }
//         }
//     }

//     // Étape 3 : Convolution
//     ly_conv: for (int f = 0; f < CONV3_FILTER_NUMBER; ++f) {
//         lx_conv: for (int c = 0; c < IMG_CHANNELS; ++c) {
//             lz_conv: for (int i = 0; i < MAXPOOL2_OUT_HEIGHT; ++i) {
//                 lw_conv: for (int j = 0; j < MAXPOOL2_OUT_WIDTH; ++j) {
//                     fixed_t conv_sum = 0;
//                     ly_k: for (int ki = 0; ki < CONV3_FILTER_HEIGHT; ++ki) {
//                         lx_k: for (int kj = 0; kj < CONV3_FILTER_WIDTH; ++kj) {
//                             conv_sum += padded_image[i + ki][j + kj][c] * Ks[ki][kj][c][f];
//                         }
//                     }
//                     output[i][j][f] += conv_sum;
//                 }
//             }
//         }
//     }

//     // Étape 4 : Activation ReLU
//     ly_relu: for (int f = 0; f < CONV3_FILTER_NUMBER; ++f) {
//         lx_relu: for (int i = 0; i < MAXPOOL2_OUT_HEIGHT; ++i) {
//             lz_relu: for (int j = 0; j < MAXPOOL2_OUT_WIDTH; ++j) {
//                 output[i][j][f] = ReLu(output[i][j][f]);
//             }
//         }
//     }
// }

// void maxpool1(
//     fixed_t input[MAXPOOL1_IN_HEIGHT][MAXPOOL1_IN_WIDTH][MAXPOOL1_IN_CHANNELS],
//     fixed_t output[MAXPOOL1_OUT_HEIGHT][MAXPOOL1_OUT_WIDTH][MAXPOOL1_OUT_CHANNELS]
// ) {
//     int c = 0;
//     int h = 0;
//     int w = 0;
//     int start_h, start_w;
//     for (c = 0; c < MAXPOOL1_IN_CHANNELS; c++) {
//         for (h = 0; h < MAXPOOL1_OUT_HEIGHT; h++) {
//             for (w = 0; w < MAXPOOL1_OUT_WIDTH; w++) {
//                 start_h = h * 2;
//                 start_w = w * 2;
//                 float max_val = -1; // <-------- plus petit possible
                
//                 if (start_h + 3 > MAXPOOL1_IN_HEIGHT) {
//                     if (start_w + 3 > MAXPOOL1_IN_WIDTH) {
//                         // region = M[start_h:y, start_w:x, c]
//                         for (int i = start_h; i < MAXPOOL1_IN_HEIGHT; i++) {
//                             for (int j = start_w; j < MAXPOOL1_IN_WIDTH; j++) {
//                                 max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
//                             }
//                         }
//                     }
//                     else {
//                         // region = M[start_h:y, start_w:start_w+3, c]
//                         for (int i = start_h; i < MAXPOOL1_IN_HEIGHT; i++) {
//                             for (int j = start_w; j < start_w + 3; j++) {
//                                 max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
//                             }
//                         }
//                     }
//                 }
//                 else if (start_w + 3 > MAXPOOL1_IN_WIDTH) {
//                     // region = M[start_h:start_h+3, start_w:x, c]
//                     for (int i = start_h; i < start_h + 3; i++) {
//                         for (int j = start_w; j < MAXPOOL1_IN_WIDTH; j++) {
//                             max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
//                         }
//                     }
//                 }
//                 else {
//                     // region = M[start_h:start_h+3, start_w:start_w+3, c]
//                     for (int i = start_h; i < start_h + 3; i++) {
//                         for (int j = start_w; j < start_w + 3; j++) {
//                             max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
//                         }
//                     }
//                 }
//                 output[h][w][c] = max_val;
//             }
//         }
//     }
// }

// void maxpool2(
//     fixed_t input[MAXPOOL2_IN_HEIGHT][MAXPOOL2_IN_WIDTH][MAXPOOL2_IN_CHANNELS],
//     fixed_t output[MAXPOOL2_OUT_HEIGHT][MAXPOOL2_OUT_WIDTH][MAXPOOL2_OUT_CHANNELS]
// ) {
//     int c = 0;
//     int h = 0;
//     int w = 0;
//     int start_h, start_w;
//     for (c = 0; c < MAXPOOL2_IN_CHANNELS; c++) {
//         for (h = 0; h < MAXPOOL2_OUT_HEIGHT; h++) {
//             for (w = 0; w < MAXPOOL2_OUT_WIDTH; w++) {
//                 start_h = h * 2;
//                 start_w = w * 2;
//                 float max_val = -1; // <-------- plus petit possible
                
//                 if (start_h + 3 > MAXPOOL2_IN_HEIGHT) {
//                     if (start_w + 3 > MAXPOOL2_IN_WIDTH) {
//                         // region = M[start_h:y, start_w:x, c]
//                         for (int i = start_h; i < MAXPOOL2_IN_HEIGHT; i++) {
//                             for (int j = start_w; j < MAXPOOL2_IN_WIDTH; j++) {
//                                 max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
//                             }
//                         }
//                     }
//                     else {
//                         // region = M[start_h:y, start_w:start_w+3, c]
//                         for (int i = start_h; i < MAXPOOL2_IN_HEIGHT; i++) {
//                             for (int j = start_w; j < start_w + 3; j++) {
//                                 max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
//                             }
//                         }
//                     }
//                 }
//                 else if (start_w + 3 > MAXPOOL2_IN_WIDTH) {
//                     // region = M[start_h:start_h+3, start_w:x, c]
//                     for (int i = start_h; i < start_h + 3; i++) {
//                         for (int j = start_w; j < MAXPOOL2_IN_WIDTH; j++) {
//                             max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
//                         }
//                     }
//                 }
//                 else {
//                     // region = M[start_h:start_h+3, start_w:start_w+3, c]
//                     for (int i = start_h; i < start_h + 3; i++) {
//                         for (int j = start_w; j < start_w + 3; j++) {
//                             max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
//                         }
//                     }
//                 }
//                 output[h][w][c] = max_val;
//             }
//         }
//     }
// }

// void maxpool3(
//     fixed_t input[MAXPOOL3_IN_HEIGHT][MAXPOOL3_IN_WIDTH][MAXPOOL3_IN_CHANNELS],
//     fixed_t output[MAXPOOL3_OUT_HEIGHT][MAXPOOL3_OUT_WIDTH][MAXPOOL3_OUT_CHANNELS]
// ) {
//     int c = 0;
//     int h = 0;
//     int w = 0;
//     int start_h, start_w;
//     for (c = 0; c < MAXPOOL3_IN_CHANNELS; c++) {
//         for (h = 0; h < MAXPOOL3_OUT_HEIGHT; h++) {
//             for (w = 0; w < MAXPOOL3_OUT_WIDTH; w++) {
//                 start_h = h * 2;
//                 start_w = w * 2;
//                 float max_val = -1; // <-------- plus petit possible
                
//                 if (start_h + 3 > MAXPOOL3_IN_HEIGHT) {
//                     if (start_w + 3 > MAXPOOL3_IN_WIDTH) {
//                         // region = M[start_h:y, start_w:x, c]
//                         for (int i = start_h; i < MAXPOOL3_IN_HEIGHT; i++) {
//                             for (int j = start_w; j < MAXPOOL3_IN_WIDTH; j++) {
//                                 max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
//                             }
//                         }
//                     }
//                     else {
//                         // region = M[start_h:y, start_w:start_w+3, c]
//                         for (int i = start_h; i < MAXPOOL3_IN_HEIGHT; i++) {
//                             for (int j = start_w; j < start_w + 3; j++) {
//                                 max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
//                             }
//                         }
//                     }
//                 }
//                 else if (start_w + 3 > MAXPOOL3_IN_WIDTH) {
//                     // region = M[start_h:start_h+3, start_w:x, c]
//                     for (int i = start_h; i < start_h + 3; i++) {
//                         for (int j = start_w; j < MAXPOOL3_IN_WIDTH; j++) {
//                             max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
//                         }
//                     }
//                 }
//                 else {
//                     // region = M[start_h:start_h+3, start_w:start_w+3, c]
//                     for (int i = start_h; i < start_h + 3; i++) {
//                         for (int j = start_w; j < start_w + 3; j++) {
//                             max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
//                         }
//                     }
//                 }
//                 output[h][w][c] = max_val;
//             }
//         }
//     }
// }

double ReLu_old(double x) {
    if (x > 0) {
	return x;
    }
    return 0.0;
}

void softmax_old(double x[10]) {
    double sum = 0.0;
    int i = 0;

    for (i = 0; i < 10; i++) {
        x[i] = expf(x[i]);
        sum += x[i];
    }

    for (int i = 0; i < 10; i++) {
        x[i] /= sum;
    }
}

void reshape_old(double input[3][3][20], double output[180]) {
    int index = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 20; k++) {
                output[index++] = input[i][j][k];
            }
        }
    }
}

void FCP_old(double M[180], double weights[180][10], double bias[10], double output[10]) {
    // NON  TESTE ATTENTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    double temp_output[10] = {0};

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 180; j++) {
            temp_output[i] += M[j] * weights[j][i];
        }
        temp_output[i] += bias[i];
    }

    softmax_old(temp_output);

    for (int i = 0; i < 10; i++) {
        output[i] = temp_output[i];
    }
}