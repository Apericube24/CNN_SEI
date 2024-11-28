#include "functions.h"
#include <math.h>

#include "virgule_fixe.h"
#include <iostream>
#include <cstdint> 


using fixed_t = FixedPoint<int32_t, 16>; 

float ReLu(float x) {
    if (x > 0) {
	return x;
    }
    return 0.0;
}

void softmax(float x[10]) {
    float sum = 0.0;
    int i = 0;

    for (i = 0; i < 10; i++) {
        x[i] = expf(x[i]);
        sum += x[i];
    }

    for (int i = 0; i < 10; i++) {
        x[i] /= sum;
    }
}

template <unit32_t IMG_H, unit32_t IMG_W, unit32_t IMG_C, unit32_t K_H, unit32_t K_W, unit32_t NUM_FILTERS>
void convolution(const FixedPoint<int32_t, 16> image[IMG_H][IMG_W][IMG_C],
                 const FixedPoint<int32_t, 16> Ks[K_H][k_W][IMG_C][NUM_FILTERS],
                 const FixedPoint<int32_t, 16> biais[NUM_FILTERS],
                 FixedPoint<int32_t, 16> output[IMG_H][IMG_W][NUM_FILTERS]) {

    constexpr unit32_t PADDING = 2;
    constexpr unit32_t PADDED_H = IMG_H + 2 * PADDING;
    constexpr unit32_t PADDED_W = IMG_W + 2 * PADDING;

    FixedPoint<int32_t, 16> padded_image[PADDED_H][PADDED_W][IMG_C]
    // padding
    for (unit32_t c = 0; c < IMG_C; ++c) {
        for (unit32_t i = 0; i < IMG_H; ++i) {
            for (unit32_t j = 0; j < IMG_W; ++j) {
                padded_image[i + PADDING][j + PADDING][c] = image[i][j][c];
            }
        }
    }

    // init biais
    for (unit32_t f = 0; f < NUM_FILTERS; ++f) {
        for (unit32_t i = 0; i < IMG_H; ++i) {
            for (unit32_t j = 0; j < IMG_W; ++j) {
                output[i][j][f] = biais[f];
            }
        }
    }

    // etape conv
    for (unit32_t f = 0; f < NUM_FILTERS; ++f) {
        for (unit32_t c = 0; c < IMG_C; ++c) {
            for (unit32_t i = 0; i < IMG_H; ++i) {
                for (unit32_t j = 0; j < IMG_W; ++j) {
                    fixed_t conv_sum = 0;
                    for (unit32_t ki = 0; ki < K_H; ++ki) {
                        for (unit32_t kj = 0; kj < K_W; ++kj) {
                            conv_sum = conv_sum + padded_image[i + ki][j + kj][c] * Ks[ki][kj][c][f];
                        }
                    }
                    output[i][j][f] = output[i][j][f] + conv_sum;
                }
            }
        }

        // etape relu
        for (unit32_t i = 0; i < IMG_H; ++i) {
            for (unit32_t j = 0; j < IMG_W; ++j) {
                output[i][j][f] = ReLu(output[i][j][f]);
            }
        }
    }
}



template <uint32_t IN_H, uint32_t IN_W, uint32_t IN_C, uint32_t OUT_H, uint32_t OUT_W>
void maxpool_1(const FixedPoint<int32_t, 16> input[IN_H][IN_W][IN_C],
               FixedPoint<int32_t, 16> output[OUT_H][OUT_W][IN_C]) {
    constexpr uint32_t POOL_SIZE = 2; // Taille du max-pooling (2x2)

    for (uint32_t c = 0; c < IN_C; ++c) {
        for (uint32_t i = 0; i < OUT_H; ++i) {
            for (uint32_t j = 0; j < OUT_W; ++j) {
                uint32_t start_i = i * POOL_SIZE;
                uint32_t start_j = j * POOL_SIZE;

                FixedPoint<int32_t, 16> max_value = input[start_i][start_j][c];
                
                for (uint32_t ki = 0; ki < POOL_SIZE; ++ki) {
                    for (uint32_t kj = 0; kj < POOL_SIZE; ++kj) {
                        uint32_t row = start_i + ki;
                        uint32_t col = start_j + kj;
                        if (row < IN_H && col < IN_W) {
                            if (input[row][col][c] > max_value) {
                                max_value = input[row][col][c];
                            }
                        }
                    }
                }

                output[i][j][c] = max_value;

            }
        }
    }
}

template <int HEIGHT, int WIDTH, int NUM_CHANNELS, int OUT_HEIGHT, int OUT_WIDTH>
void maxpool_3x3(float input[HEIGHT][WIDTH][NUM_CHANNELS], float output[OUT_HEIGHT][OUT_WIDTH][NUM_CHANNELS]) {
    int c = 0;
    int h = 0;
    int w = 0;
    int start_h, start_w;
    for (c = 0; c < NUM_CHANNELS; c++) {
        for (h = 0; h < OUT_HEIGHT; h++) {
            for (w = 0; w < OUT_WIDTH; w++) {
                start_h = h * 2;
                start_w = w * 2;
                float max_val = -FLT_MAX; // <-------- plus petit possible
                
                if (start_h + 3 > OUT_HEIGHT) {
                    if (start_w + 3 > OUT_WIDTH) {
                        // region = M[start_h:y, start_w:x, c]
                        for (int i = start_h; i < OUT_HEIGHT; i++) {
                            for (int j = start_w; j < OUT_WIDTH; j++) {
                                max_val = (M[i][j][c] > max_val) ? M[i][j][c] : max_val;
                            }
                        }
                    }
                    else {
                        // region = M[start_h:y, start_w:start_w+3, c]
                        for (int i = start_h; i < OUT_HEIGHT; i++) {
                            for (int j = start_w; j < start_w + 3; j++) {
                                max_val = (M[i][j][c] > max_val) ? M[i][j][c] : max_val;
                            }
                        }
                    }
                }
                else if (start_w + 3 > OUT_WIDTH) {
                    // region = M[start_h:start_h+3, start_w:x, c]
                    for (int i = start_h; i < start_h + 3; i++) {
                        for (int j = start_w; j < OUT_WIDTH; j++) {
                            max_val = (M[i][j][c] > max_val) ? M[i][j][c] : max_val;
                        }
                    }
                }
                else {
                    // region = M[start_h:start_h+3, start_w:start_w+3, c]
                    for (int i = start_h; i < start_h + 3; i++) {
                        for (int j = start_w; j < start_w + 3; j++) {
                            max_val = (M[i][j][c] > max_val) ? M[i][j][c] : max_val;
                        }
                    }
                }
                output[h][w][c] = max_val;
            }
        }
    }
}