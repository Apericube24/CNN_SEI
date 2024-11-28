#include "functions.h"
#include <math.h>

#include "virgule_fixe.h"

// Typedef pour fixer la précision
using fixed_t = FixedPoint<int32_t, 16>; //16 bit de fraction j'ai fait a la va vite pas sûr que ça soit OK

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

template <size_t IMG_H, size_t IMG_W, size_t IMG_C, size_t K_H, size_t K_W, size_t NUM_FILTERS>
void convolution(const std::array<std::array<std::array<fixed_t, IMG_C>, IMG_W>, IMG_H>& image,
                 const std::array<std::array<std::array<std::array<fixed_t, NUM_FILTERS>, IMG_C>, K_W>, K_H>& Ks,
                 const std::array<fixed_t, NUM_FILTERS>& biais,
                 std::array<std::array<std::array<fixed_t, NUM_FILTERS>, IMG_W>, IMG_H>& output) {
    constexpr size_t PADDING = 2;
    constexpr size_t PADDED_H = IMG_H + 2 * PADDING;
    constexpr size_t PADDED_W = IMG_W + 2 * PADDING;

    std::array<std::array<std::array<fixed_t, IMG_C>, PADDED_W>, PADDED_H> padded_image = {};

    // padding
    for (size_t c = 0; c < IMG_C; ++c) {
        for (size_t i = 0; i < IMG_H; ++i) {
            for (size_t j = 0; j < IMG_W; ++j) {
                padded_image[i + PADDING][j + PADDING][c] = image[i][j][c];
            }
        }
    }

    // init biais
    for (size_t f = 0; f < NUM_FILTERS; ++f) {
        for (size_t i = 0; i < IMG_H; ++i) {
            for (size_t j = 0; j < IMG_W; ++j) {
                output[i][j][f] = biais[f];
            }
        }
    }

    // etape conv
    for (size_t f = 0; f < NUM_FILTERS; ++f) {
        for (size_t c = 0; c < IMG_C; ++c) {
            for (size_t i = 0; i < IMG_H; ++i) {
                for (size_t j = 0; j < IMG_W; ++j) {
                    fixed_t conv_sum = 0;
                    for (size_t ki = 0; ki < K_H; ++ki) {
                        for (size_t kj = 0; kj < K_W; ++kj) {
                            conv_sum = conv_sum + padded_image[i + ki][j + kj][c] * Ks[ki][kj][c][f];
                        }
                    }
                    output[i][j][f] = output[i][j][f] + conv_sum;
                }
            }
        }

        // etape relu
        for (size_t i = 0; i < IMG_H; ++i) {
            for (size_t j = 0; j < IMG_W; ++j) {
                output[i][j][f] = ReLu(output[i][j][f]);
            }
        }
    }
}


template <size_t IN_H, size_t IN_W, size_t IN_C, size_t OUT_H, size_t OUT_W>
void maxpool(const std::array<std::array<std::array<fixed_t, IN_C>, IN_W>, IN_H>& input,
               std::array<std::array<std::array<fixed_t, IN_C>, OUT_W>, OUT_H>& output) {
    constexpr size_t POOL_SIZE = 2; // Taille du max-pooling (2x2)

    for (size_t c = 0; c < IN_C; ++c) { 
        for (size_t i = 0; i < OUT_H; ++i) {
            for (size_t j = 0; j < OUT_W; ++j) {
                // Trouver la région 2x2 sur laquelle effectuer le max
                size_t start_i = i * POOL_SIZE;
                size_t start_j = j * POOL_SIZE;

                
                fixed_t max_value = input[start_i][start_j][c];

                // Parcours de la région 2x2
                for (size_t ki = 0; ki < POOL_SIZE; ++ki) {
                    for (size_t kj = 0; kj < POOL_SIZE; ++kj) {
                        size_t row = start_i + ki;
                        size_t col = start_j + kj;
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
