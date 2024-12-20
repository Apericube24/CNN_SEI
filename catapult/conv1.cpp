#include "functions.hpp"
#include "ac_fixed.h"

// defining values
#define IMG_HEIGHT 24
#define IMG_WIDTH 24
#define IMG_CHANNELS 3
#define IMG_SIZE (IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS)

#define CONV1_FILTER_HEIGHT 3
#define CONV1_FILTER_WIDTH 3
#define CONV1_FILTER_CHANNELS 3
#define CONV1_FILTER_NUMBER 64
#define CONV1_SIZE (CONV1_FILTER_HEIGHT * CONV1_FILTER_WIDTH * CONV1_FILTER_CHANNELS * CONV1_FILTER_NUMBER)
#define CONV1_BIAS_NUMBER 64

#define MAXPOOL1_IN_HEIGHT 24
#define MAXPOOL1_IN_WIDTH 24
#define MAXPOOL1_IN_CHANNELS 64
#define MAXPOOL1_IN_SIZE (MAXPOOL1_IN_HEIGHT * MAXPOOL1_IN_WIDTH * MAXPOOL1_IN_CHANNELS)

#define addr2D(x, y, SIZE_Y) (x)*(SIZE_Y) + (y)
#define addr3D(x, y, z, SIZE_Y, SIZE_Z) (x)*(SIZE_Y)*(SIZE_Z) + (y)*(SIZE_Z) + (z)
#define addr4D(x, y, z, t, SIZE_Y, SIZE_Z, SIZE_T) (x)*(SIZE_Y)*(SIZE_Z)*(SIZE_T) + (y)*(SIZE_Z)*(SIZE_T) + (z)*(SIZE_T) + (t)

ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> ReLu(ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> x) {
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> result;
    if (x > 0) {
        result = x;
    }
    else {
        result = 0;
    }
    return result;
}

void convolution1(
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> image[IMG_SIZE],
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> Ks[CONV1_SIZE],
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> biais[CONV1_BIAS_NUMBER],
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> output[MAXPOOL1_IN_SIZE]
) {
    int PADDING = 1;
    int PADDED_H = IMG_HEIGHT + 2 * PADDING;
    int PADDED_W = IMG_WIDTH + 2 * PADDING;

    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> padded_image[PADDED_H * PADDED_W * IMG_CHANNELS];

    int c, i, j, f, ki, kj;
    // etape 1 : Padding
    ly_pad: for (c= 0; c < IMG_CHANNELS; ++c) {
        lx_pad: for (i = 0; i < PADDED_H; ++i) {
            lz_pad: for (j = 0; j < PADDED_W; ++j) {
                if (i < PADDING || j < PADDING || i >= PADDED_H - PADDING || j >= PADDED_W - PADDING) {
		            padded_image[addr3D(i, j, c, PADDED_H, IMG_CHANNELS)] = 0.0;
                } else {
                    padded_image[addr3D(i, j, c, PADDED_H, IMG_CHANNELS)]= image[addr3D(i-PADDING, j-PADDING, c, IMG_HEIGHT, IMG_CHANNELS)];
                }
            }
        }
    }

    // etape 2 : Initialisation des sorties avec les biais
    ly_init: for (f = 0; f < CONV1_FILTER_NUMBER; ++f) {
        lx_init: for (i = 0; i < IMG_HEIGHT; ++i) {
            lz_init: for (j = 0; j < IMG_WIDTH; ++j) {
                output[addr3D(i, j, f, MAXPOOL1_IN_HEIGHT, CONV1_FILTER_NUMBER)] = biais[f];
            }
        }
    }

    // etape 3 : Convolution
    ly_conv: for (f = 0; f < CONV1_FILTER_NUMBER; ++f) {
        lx_conv: for (c = 0; c < IMG_CHANNELS; ++c) {
            lz_conv: for (i = 0; i < IMG_HEIGHT; ++i) {
                lw_conv: for (j = 0; j < IMG_WIDTH; ++j) {
                    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> conv_sum = 0;
                    ly_k: for (ki = 0; ki < CONV1_FILTER_HEIGHT; ++ki) {
                        lx_k: for (kj = 0; kj < CONV1_FILTER_WIDTH; ++kj) {
			                conv_sum += padded_image[addr3D(i+ki, j+kj, c,PADDED_H, IMG_CHANNELS)]* Ks[addr4D(ki, kj, c, f, CONV1_FILTER_HEIGHT, CONV1_FILTER_CHANNELS, CONV1_FILTER_NUMBER)];
                        }
                    }
                    output[addr3D(i, j, f, MAXPOOL1_IN_HEIGHT, CONV1_FILTER_NUMBER)] += conv_sum;
                }
            }
        }
    }

    // etape 4 : Activation ReLU
    ly_relu: for (f = 0; f < CONV1_FILTER_NUMBER; ++f) {
        lx_relu: for (i = 0; i < IMG_HEIGHT; ++i) {
            lz_relu: for (j = 0; j < IMG_WIDTH; ++j) {
                output[addr3D(i, j, f, MAXPOOL1_IN_HEIGHT, CONV1_FILTER_NUMBER)] = ReLu(output[addr3D(i, j, f, MAXPOOL1_IN_HEIGHT, CONV1_FILTER_NUMBER)]);
            }
        }
    }
}