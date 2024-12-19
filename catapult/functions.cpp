#include "functions.hpp"
#include <math.h>
#include <cmath>

#include <iostream>
#include <cstdint>
#include "ac_fixed.h"

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

void reshape(ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> input[3][3][20], ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> output[180]) {
    int index = 0;
    int i, j, k;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            for (k = 0; k < 20; k++) {
                output[index++] = input[i][j][k];
            }
        }
    }
}

void FCP(ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> M[180], ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> weights[180][10], ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> bias[10], ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> output[10]) {
    // NON  TESTE ATTENTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> temp_output[10] = {0};
    int i, j;
    for (i = 0; i < 10; i++) {
        for (j = 0; j < 180; j++) {
            temp_output[i] += M[j] * weights[j][i];
        }
        temp_output[i] += bias[i];
    }
}

void convolution1(
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> image[IMG_HEIGHT][IMG_WIDTH][IMG_CHANNELS],
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> Ks[CONV1_FILTER_HEIGHT][CONV1_FILTER_WIDTH][CONV1_FILTER_CHANNELS][CONV1_FILTER_NUMBER],
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> biais[CONV1_BIAS_NUMBER],
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> output[MAXPOOL1_IN_HEIGHT][MAXPOOL1_IN_WIDTH][MAXPOOL1_IN_CHANNELS]
) {
    const int PADDING = 1;
    const int PADDED_H = IMG_HEIGHT + 2 * PADDING;
    const int PADDED_W = IMG_WIDTH + 2 * PADDING;

    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> padded_image[PADDED_H][PADDED_W][IMG_CHANNELS];

    int c, i, j, f, ki, kj;
    // Étape 1 : Padding
    ly_pad: for (c= 0; c < IMG_CHANNELS; ++c) {
        lx_pad: for (i = 0; i < PADDED_H; ++i) {
            lz_pad: for (j = 0; j < PADDED_W; ++j) {
                if (i < PADDING || j < PADDING || i >= PADDED_H - PADDING || j >= PADDED_W - PADDING) {
                    padded_image[i][j][c] = 0.0;
                } else {
                    padded_image[i][j][c] = image[i - PADDING][j - PADDING][c];
                }
            }
        }
    }

    // Étape 2 : Initialisation des sorties avec les biais
    ly_init: for (f = 0; f < CONV1_FILTER_NUMBER; ++f) {
        lx_init: for (i = 0; i < IMG_HEIGHT; ++i) {
            lz_init: for (j = 0; j < IMG_WIDTH; ++j) {
                output[i][j][f] = biais[f];
            }
        }
    }

    // Étape 3 : Convolution
    ly_conv: for (f = 0; f < CONV1_FILTER_NUMBER; ++f) {
        lx_conv: for (c = 0; c < IMG_CHANNELS; ++c) {
            lz_conv: for (i = 0; i < IMG_HEIGHT; ++i) {
                lw_conv: for (j = 0; j < IMG_WIDTH; ++j) {
                    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> conv_sum = 0;
                    ly_k: for (ki = 0; ki < CONV1_FILTER_HEIGHT; ++ki) {
                        lx_k: for (kj = 0; kj < CONV1_FILTER_WIDTH; ++kj) {
                            conv_sum += padded_image[i + ki][j + kj][c] * Ks[ki][kj][c][f];
                        }
                    }
                    output[i][j][f] += conv_sum;
                }
            }
        }
    }

    // Étape 4 : Activation ReLU
    ly_relu: for (f = 0; f < CONV1_FILTER_NUMBER; ++f) {
        lx_relu: for (i = 0; i < IMG_HEIGHT; ++i) {
            lz_relu: for (j = 0; j < IMG_WIDTH; ++j) {
                output[i][j][f] = ReLu(output[i][j][f]);
            }
        }
    }
}

void convolution2(
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> image[MAXPOOL1_OUT_HEIGHT][MAXPOOL1_OUT_WIDTH][MAXPOOL1_OUT_CHANNELS],
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> Ks[CONV2_FILTER_HEIGHT][CONV2_FILTER_WIDTH][CONV2_FILTER_CHANNELS][CONV2_FILTER_NUMBER],
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> biais[CONV2_BIAS_NUMBER],
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> output[MAXPOOL2_IN_HEIGHT][MAXPOOL2_IN_WIDTH][MAXPOOL2_IN_CHANNELS]
) {
    const int PADDING = 1;
    const int PADDED_H = MAXPOOL1_OUT_HEIGHT + 2 * PADDING;
    const int PADDED_W = MAXPOOL1_OUT_WIDTH + 2 * PADDING;

    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> padded_image[PADDED_H][PADDED_W][IMG_CHANNELS];

    int c, i, j, f, ki, kj;
    // Étape 1 : Padding
    ly_pad: for (c = 0; c < MAXPOOL1_OUT_CHANNELS; ++c) {
        lx_pad: for (i = 0; i < PADDED_H; ++i) {
            lz_pad: for (j = 0; j < PADDED_W; ++j) {
                if (i < PADDING || j < PADDING || i >= PADDED_H - PADDING || j >= PADDED_W - PADDING) {
                    padded_image[i][j][c] = 0.0;
                } else {
                    padded_image[i][j][c] = image[i - PADDING][j - PADDING][c];
                }
            }
        }
    }

    // Étape 2 : Initialisation des sorties avec les biais
    ly_init: for (f = 0; f < CONV2_FILTER_NUMBER; ++f) {
        lx_init: for (i = 0; i < MAXPOOL1_OUT_HEIGHT; ++i) {
            lz_init: for (j = 0; j < MAXPOOL1_OUT_WIDTH; ++j) {
                output[i][j][f] = biais[f];
            }
        }
    }

    // Étape 3 : Convolution
    ly_conv: for (f = 0; f < CONV2_FILTER_NUMBER; ++f) {
        lx_conv: for (c = 0; c < IMG_CHANNELS; ++c) {
            lz_conv: for (i = 0; i < MAXPOOL1_OUT_HEIGHT; ++i) {
                lw_conv: for (j = 0; j < MAXPOOL1_OUT_WIDTH; ++j) {
                    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> conv_sum = 0;
                    ly_k: for (ki = 0; ki < CONV2_FILTER_HEIGHT; ++ki) {
                        lx_k: for (kj = 0; kj < CONV2_FILTER_WIDTH; ++kj) {
                            conv_sum += padded_image[i + ki][j + kj][c] * Ks[ki][kj][c][f];
                        }
                    }
                    output[i][j][f] += conv_sum;
                }
            }
        }
    }

    // Étape 4 : Activation ReLU
    ly_relu: for (f = 0; f < CONV2_FILTER_NUMBER; ++f) {
        lx_relu: for (i = 0; i < MAXPOOL1_OUT_HEIGHT; ++i) {
            lz_relu: for (j = 0; j < MAXPOOL1_OUT_WIDTH; ++j) {
                output[i][j][f] = ReLu(output[i][j][f]);
            }
        }
    }
}

void convolution3(
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> image[MAXPOOL2_OUT_HEIGHT][MAXPOOL2_OUT_WIDTH][MAXPOOL2_OUT_CHANNELS],
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> Ks[CONV3_FILTER_HEIGHT][CONV3_FILTER_WIDTH][CONV3_FILTER_CHANNELS][CONV3_FILTER_NUMBER],
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> biais[CONV3_BIAS_NUMBER],
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> output[MAXPOOL3_IN_HEIGHT][MAXPOOL3_IN_WIDTH][MAXPOOL3_IN_CHANNELS]
) {
    const int PADDING = 1;
    const int PADDED_H = MAXPOOL2_OUT_HEIGHT + 2 * PADDING;
    const int PADDED_W = MAXPOOL2_OUT_WIDTH + 2 * PADDING;

    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> padded_image[PADDED_H][PADDED_W][IMG_CHANNELS];

    int c, i, j, f, ki, kj;
    // Étape 1 : Padding
    ly_pad: for (c = 0; c < MAXPOOL2_OUT_CHANNELS; ++c) {
        lx_pad: for (i = 0; i < PADDED_H; ++i) {
            lz_pad: for (j = 0; j < PADDED_W; ++j) {
                if (i < PADDING || j < PADDING || i >= PADDED_H - PADDING || j >= PADDED_W - PADDING) {
                    padded_image[i][j][c] = 0.0;
                } else {
                    padded_image[i][j][c] = image[i - PADDING][j - PADDING][c];
                }
            }
        }
    }

    // Étape 2 : Initialisation des sorties avec les biais
    ly_init: for (f = 0; f < CONV3_FILTER_NUMBER; ++f) {
        lx_init: for (i = 0; i < MAXPOOL2_OUT_HEIGHT; ++i) {
            lz_init: for (j = 0; j < MAXPOOL2_OUT_WIDTH; ++j) {
                output[i][j][f] = biais[f];
            }
        }
    }

    // Étape 3 : Convolution
    ly_conv: for (f = 0; f < CONV3_FILTER_NUMBER; ++f) {
        lx_conv: for (c = 0; c < IMG_CHANNELS; ++c) {
            lz_conv: for (i = 0; i < MAXPOOL2_OUT_HEIGHT; ++i) {
                lw_conv: for (j = 0; j < MAXPOOL2_OUT_WIDTH; ++j) {
                    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> conv_sum = 0;
                    ly_k: for (ki = 0; ki < CONV3_FILTER_HEIGHT; ++ki) {
                        lx_k: for (kj = 0; kj < CONV3_FILTER_WIDTH; ++kj) {
                            conv_sum += padded_image[i + ki][j + kj][c] * Ks[ki][kj][c][f];
                        }
                    }
                    output[i][j][f] += conv_sum;
                }
            }
        }
    }

    // Étape 4 : Activation ReLU
    ly_relu: for (f = 0; f < CONV3_FILTER_NUMBER; ++f) {
        lx_relu: for (i = 0; i < MAXPOOL2_OUT_HEIGHT; ++i) {
            lz_relu: for (j = 0; j < MAXPOOL2_OUT_WIDTH; ++j) {
                output[i][j][f] = ReLu(output[i][j][f]);
            }
        }
    }
}

void maxpool1(
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> input[MAXPOOL1_IN_HEIGHT][MAXPOOL1_IN_WIDTH][MAXPOOL1_IN_CHANNELS],
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> output[MAXPOOL1_OUT_HEIGHT][MAXPOOL1_OUT_WIDTH][MAXPOOL1_OUT_CHANNELS]
) {
    int c = 0;
    int h = 0;
    int w = 0;
    int i, j;
    int start_h, start_w;
    for (c = 0; c < MAXPOOL1_IN_CHANNELS; c++) {
        for (h = 0; h < MAXPOOL1_OUT_HEIGHT; h++) {
            for (w = 0; w < MAXPOOL1_OUT_WIDTH; w++) {
                start_h = h * 2;
                start_w = w * 2;
                ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> max_val = -1; // <-------- plus petit possible
                
                if (start_h + 3 > MAXPOOL1_IN_HEIGHT) {
                    if (start_w + 3 > MAXPOOL1_IN_WIDTH) {
                        // region = M[start_h:y, start_w:x, c]
                        for (i = start_h; i < MAXPOOL1_IN_HEIGHT; i++) {
                            for (j = start_w; j < MAXPOOL1_IN_WIDTH; j++) {
                                max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
                            }
                        }
                    }
                    else {
                        // region = M[start_h:y, start_w:start_w+3, c]
                        for (i = start_h; i < MAXPOOL1_IN_HEIGHT; i++) {
                            for (j = start_w; j < start_w + 3; j++) {
                                max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
                            }
                        }
                    }
                }
                else if (start_w + 3 > MAXPOOL1_IN_WIDTH) {
                    // region = M[start_h:start_h+3, start_w:x, c]
                    for (i = start_h; i < start_h + 3; i++) {
                        for (j = start_w; j < MAXPOOL1_IN_WIDTH; j++) {
                            max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
                        }
                    }
                }
                else {
                    // region = M[start_h:start_h+3, start_w:start_w+3, c]
                    for (i = start_h; i < start_h + 3; i++) {
                        for (j = start_w; j < start_w + 3; j++) {
                            max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
                        }
                    }
                }
                output[h][w][c] = max_val;
            }
        }
    }
}

void maxpool2(
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> input[MAXPOOL2_IN_HEIGHT][MAXPOOL2_IN_WIDTH][MAXPOOL2_IN_CHANNELS],
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> output[MAXPOOL2_OUT_HEIGHT][MAXPOOL2_OUT_WIDTH][MAXPOOL2_OUT_CHANNELS]
) {
    int c = 0;
    int h = 0;
    int w = 0;
    int i, j;
    int start_h, start_w;
    for (c = 0; c < MAXPOOL2_IN_CHANNELS; c++) {
        for (h = 0; h < MAXPOOL2_OUT_HEIGHT; h++) {
            for (w = 0; w < MAXPOOL2_OUT_WIDTH; w++) {
                start_h = h * 2;
                start_w = w * 2;
                ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> max_val = -1; // <-------- plus petit possible
                
                if (start_h + 3 > MAXPOOL2_IN_HEIGHT) {
                    if (start_w + 3 > MAXPOOL2_IN_WIDTH) {
                        // region = M[start_h:y, start_w:x, c]
                        for (i = start_h; i < MAXPOOL2_IN_HEIGHT; i++) {
                            for (j = start_w; j < MAXPOOL2_IN_WIDTH; j++) {
                                max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
                            }
                        }
                    }
                    else {
                        // region = M[start_h:y, start_w:start_w+3, c]
                        for (i = start_h; i < MAXPOOL2_IN_HEIGHT; i++) {
                            for (j = start_w; j < start_w + 3; j++) {
                                max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
                            }
                        }
                    }
                }
                else if (start_w + 3 > MAXPOOL2_IN_WIDTH) {
                    // region = M[start_h:start_h+3, start_w:x, c]
                    for (i = start_h; i < start_h + 3; i++) {
                        for (j = start_w; j < MAXPOOL2_IN_WIDTH; j++) {
                            max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
                        }
                    }
                }
                else {
                    // region = M[start_h:start_h+3, start_w:start_w+3, c]
                    for (i = start_h; i < start_h + 3; i++) {
                        for (j = start_w; j < start_w + 3; j++) {
                            max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
                        }
                    }
                }
                output[h][w][c] = max_val;
            }
        }
    }
}

void maxpool3(
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> input[MAXPOOL3_IN_HEIGHT][MAXPOOL3_IN_WIDTH][MAXPOOL3_IN_CHANNELS],
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> output[MAXPOOL3_OUT_HEIGHT][MAXPOOL3_OUT_WIDTH][MAXPOOL3_OUT_CHANNELS]
) {
    int c = 0;
    int h = 0;
    int w = 0;
    int i, j;
    int start_h, start_w;
    for (c = 0; c < MAXPOOL3_IN_CHANNELS; c++) {
        for (h = 0; h < MAXPOOL3_OUT_HEIGHT; h++) {
            for (w = 0; w < MAXPOOL3_OUT_WIDTH; w++) {
                start_h = h * 2;
                start_w = w * 2;
                ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> max_val = -1; // <-------- plus petit possible
                
                if (start_h + 3 > MAXPOOL3_IN_HEIGHT) {
                    if (start_w + 3 > MAXPOOL3_IN_WIDTH) {
                        // region = M[start_h:y, start_w:x, c]
                        for (i = start_h; i < MAXPOOL3_IN_HEIGHT; i++) {
                            for (j = start_w; j < MAXPOOL3_IN_WIDTH; j++) {
                                max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
                            }
                        }
                    }
                    else {
                        // region = M[start_h:y, start_w:start_w+3, c]
                        for (i = start_h; i < MAXPOOL3_IN_HEIGHT; i++) {
                            for (j = start_w; j < start_w + 3; j++) {
                                max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
                            }
                        }
                    }
                }
                else if (start_w + 3 > MAXPOOL3_IN_WIDTH) {
                    // region = M[start_h:start_h+3, start_w:x, c]
                    for (i = start_h; i < start_h + 3; i++) {
                        for (j = start_w; j < MAXPOOL3_IN_WIDTH; j++) {
                            max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
                        }
                    }
                }
                else {
                    // region = M[start_h:start_h+3, start_w:start_w+3, c]
                    for (i = start_h; i < start_h + 3; i++) {
                        for (j = start_w; j < start_w + 3; j++) {
                            max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
                        }
                    }
                }
                output[h][w][c] = max_val;
            }
        }
    }
}