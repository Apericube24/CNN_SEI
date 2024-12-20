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

#define MAXPOOL1_OUT_HEIGHT 12
#define MAXPOOL1_OUT_WIDTH 12
#define MAXPOOL1_OUT_CHANNELS 64
#define MAXPOOL1_OUT_SIZE (MAXPOOL1_OUT_HEIGHT * MAXPOOL1_OUT_WIDTH * MAXPOOL1_OUT_CHANNELS)


#define CONV2_FILTER_HEIGHT 3
#define CONV2_FILTER_WIDTH 3
#define CONV2_FILTER_CHANNELS 64
#define CONV2_FILTER_NUMBER 32
#define CONV2_SIZE (CONV2_FILTER_HEIGHT * CONV2_FILTER_WIDTH * CONV2_FILTER_CHANNELS * CONV2_FILTER_NUMBER)
#define CONV2_BIAS_NUMBER 32

#define MAXPOOL2_IN_HEIGHT 12
#define MAXPOOL2_IN_WIDTH 12
#define MAXPOOL2_IN_CHANNELS 32
#define MAXPOOL2_IN_SIZE (MAXPOOL2_IN_HEIGHT * MAXPOOL2_IN_WIDTH * MAXPOOL2_IN_CHANNELS)

#define MAXPOOL2_OUT_HEIGHT 6
#define MAXPOOL2_OUT_WIDTH 6
#define MAXPOOL2_OUT_CHANNELS 32
#define MAXPOOL2_OUT_SIZE (MAXPOOL2_OUT_HEIGHT * MAXPOOL2_OUT_WIDTH * MAXPOOL2_OUT_CHANNELS)


#define CONV3_FILTER_HEIGHT 3
#define CONV3_FILTER_WIDTH 3
#define CONV3_FILTER_CHANNELS 32
#define CONV3_FILTER_NUMBER 20
#define CONV3_SIZE (CONV3_FILTER_HEIGHT * CONV3_FILTER_WIDTH * CONV3_FILTER_CHANNELS * CONV3_FILTER_NUMBER)
#define CONV3_BIAS_NUMBER 20

#define MAXPOOL3_IN_HEIGHT 6
#define MAXPOOL3_IN_WIDTH 6
#define MAXPOOL3_IN_CHANNELS 20
#define MAXPOOL3_IN_SIZE (MAXPOOL3_IN_HEIGHT * MAXPOOL3_IN_WIDTH * MAXPOOL3_IN_CHANNELS)

#define MAXPOOL3_OUT_HEIGHT 3
#define MAXPOOL3_OUT_WIDTH 3
#define MAXPOOL3_OUT_CHANNELS 20
#define MAXPOOL3_OUT_SIZE (MAXPOOL3_OUT_HEIGHT * MAXPOOL3_OUT_WIDTH * MAXPOOL3_OUT_CHANNELS)


#define MAXPOOL_FILTER_WIDTH 3
#define MAXPOOL_FILTER_DEPTH 3


FixedPoint ReLu(FixedPoint x) {
    FixedPoint result;
    if (x > 0) {
        result = x;
    }
    else {
        result = 0;
    }
    return result;
}

void reshape(FixedPoint input[3*3*20], FixedPoint output[180]) {
    int index = 0;
    int i, j, k;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            for (k = 0; k < 20; k++) {
                output[index++] = input[addr3D(i, j, k, 3, 20)];
            }
        }
    }
}

void FCP(FixedPoint M[180], FixedPoint weights[180*10], FixedPoint bias[10], FixedPoint output[10]) {
    // NON  TESTE ATTENTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    int i, j;
    for (i = 0; i < 10; i++) {
        for (j = 0; j < 180; j++) {
            output[i] += M[j] * weights[addr2D(j, i, 10)];
        }
        output[i] += bias[i];
    }
}

void convolution1(
    FixedPoint image[IMG_SIZE],
    FixedPoint Ks[CONV1_SIZE],
    FixedPoint biais[CONV1_BIAS_NUMBER],
    FixedPoint output[MAXPOOL1_IN_SIZE],
    FixedPoint padded_image[26*26*3]
) {
    int PADDING = 1;
    int PADDED_H = IMG_HEIGHT + 2 * PADDING;
    int PADDED_W = IMG_WIDTH + 2 * PADDING;

    // FixedPoint padded_image[PADDED_H * PADDED_W * IMG_CHANNELS];

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
                    FixedPoint conv_sum = 0;
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

void convolution2(
    FixedPoint image[MAXPOOL1_OUT_SIZE],
    FixedPoint Ks[CONV2_SIZE],
    FixedPoint biais[CONV2_BIAS_NUMBER],
    FixedPoint output[MAXPOOL2_IN_SIZE],
    FixedPoint padded_image[14*14*64]
) {
    const int PADDING = 1;
    const int PADDED_H = MAXPOOL1_OUT_HEIGHT + 2 * PADDING;
    const int PADDED_W = MAXPOOL1_OUT_WIDTH + 2 * PADDING;

    // FixedPoint padded_image[PADDED_H * PADDED_W * IMG_CHANNELS];

    int c, i, j, f, ki, kj;
    // Étape 1 : Padding
    ly_pad: for (c = 0; c < MAXPOOL1_OUT_CHANNELS; ++c) {
        lx_pad: for (i = 0; i < PADDED_H; ++i) {
            lz_pad: for (j = 0; j < PADDED_W; ++j) {
                if (i < PADDING || j < PADDING || i >= PADDED_H - PADDING || j >= PADDED_W - PADDING) {
                    padded_image[addr3D(i, j, c, PADDED_H, MAXPOOL1_OUT_CHANNELS)] = 0.0;
                } else {
                    padded_image[addr3D(i, j, c, PADDED_H, MAXPOOL1_OUT_CHANNELS)] = image[addr3D(i-PADDING, j-PADDING, c, MAXPOOL1_OUT_HEIGHT, MAXPOOL1_OUT_CHANNELS)];
                }
            }
        }
    }

    // Étape 2 : Initialisation des sorties avec les biais
    ly_init: for (f = 0; f < CONV2_FILTER_NUMBER; ++f) {
        lx_init: for (i = 0; i < MAXPOOL1_OUT_HEIGHT; ++i) {
            lz_init: for (j = 0; j < MAXPOOL1_OUT_WIDTH; ++j) {
                output[addr3D(i, j, f, MAXPOOL2_IN_HEIGHT, CONV2_FILTER_NUMBER)] = biais[f];
            }
        }
    }

    // Étape 3 : Convolution
    ly_conv: for (f = 0; f < CONV2_FILTER_NUMBER; ++f) {
        lx_conv: for (c = 0; c < MAXPOOL1_OUT_CHANNELS; ++c) {
            lz_conv: for (i = 0; i < MAXPOOL1_OUT_HEIGHT; ++i) {
                lw_conv: for (j = 0; j < MAXPOOL1_OUT_WIDTH; ++j) {
                    FixedPoint conv_sum = 0;
                    ly_k: for (ki = 0; ki < CONV2_FILTER_HEIGHT; ++ki) {
                        lx_k: for (kj = 0; kj < CONV2_FILTER_WIDTH; ++kj) {
                            conv_sum += padded_image[addr3D(i+ki, j+kj, c, PADDED_H, MAXPOOL1_OUT_CHANNELS)]* Ks[addr4D(ki, kj, c, f, CONV2_FILTER_HEIGHT, CONV2_FILTER_CHANNELS, CONV2_FILTER_NUMBER)];
                        }
                    }
                    output[addr3D(i, j, f, MAXPOOL2_IN_HEIGHT, CONV2_FILTER_NUMBER)] += conv_sum;
                }
            }
        }
    }

    // Étape 4 : Activation ReLU
    ly_relu: for (f = 0; f < CONV2_FILTER_NUMBER; ++f) {
        lx_relu: for (i = 0; i < MAXPOOL1_OUT_HEIGHT; ++i) {
            lz_relu: for (j = 0; j < MAXPOOL1_OUT_WIDTH; ++j) {
                output[addr3D(i, j, f, MAXPOOL2_IN_HEIGHT, CONV2_FILTER_NUMBER)] = ReLu(output[addr3D(i, j, f, MAXPOOL2_IN_HEIGHT, CONV2_FILTER_NUMBER)]);
            }
        }
    }
}

void convolution3(
    FixedPoint image[MAXPOOL2_OUT_SIZE],
    FixedPoint Ks[CONV3_SIZE],
    FixedPoint biais[CONV3_BIAS_NUMBER],
    FixedPoint output[MAXPOOL3_IN_SIZE],
    FixedPoint padded_image[8*8*32]
) {
    const int PADDING = 1;
    const int PADDED_H = MAXPOOL2_OUT_HEIGHT + 2 * PADDING;
    const int PADDED_W = MAXPOOL2_OUT_WIDTH + 2 * PADDING;

    // FixedPoint padded_image[PADDED_H * PADDED_W * IMG_CHANNELS];

    int c, i, j, f, ki, kj;
    // Étape 1 : Padding
    ly_pad: for (c = 0; c < MAXPOOL2_OUT_CHANNELS; ++c) {
        lx_pad: for (i = 0; i < PADDED_H; ++i) {
            lz_pad: for (j = 0; j < PADDED_W; ++j) {
                if (i < PADDING || j < PADDING || i >= PADDED_H - PADDING || j >= PADDED_W - PADDING) {
                    padded_image[addr3D(i, j, c, PADDED_H, MAXPOOL2_OUT_CHANNELS)] = 0.0;
                } else {
                    padded_image[addr3D(i, j, c, PADDED_H, MAXPOOL2_OUT_CHANNELS)] = image[addr3D(i-PADDING, j-PADDING, c, MAXPOOL2_OUT_HEIGHT, MAXPOOL2_OUT_CHANNELS)];
                }
            }
        }
    }

    // Étape 2 : Initialisation des sorties avec les biais
    ly_init: for (f = 0; f < CONV3_FILTER_NUMBER; ++f) {
        lx_init: for (i = 0; i < MAXPOOL2_OUT_HEIGHT; ++i) {
            lz_init: for (j = 0; j < MAXPOOL2_OUT_WIDTH; ++j) {
                output[addr3D(i, j, f, MAXPOOL3_IN_HEIGHT, CONV3_FILTER_NUMBER)] = biais[f];
            }
        }
    }

    // Étape 3 : Convolution
    ly_conv: for (f = 0; f < CONV3_FILTER_NUMBER; ++f) {
        lx_conv: for (c = 0; c < MAXPOOL2_OUT_CHANNELS; ++c) {
            lz_conv: for (i = 0; i < MAXPOOL2_OUT_HEIGHT; ++i) {
                lw_conv: for (j = 0; j < MAXPOOL2_OUT_WIDTH; ++j) {
                    FixedPoint conv_sum = 0;
                    ly_k: for (ki = 0; ki < CONV3_FILTER_HEIGHT; ++ki) {
                        lx_k: for (kj = 0; kj < CONV3_FILTER_WIDTH; ++kj) {
                            conv_sum += padded_image[addr3D(i+ki, j+kj, c, PADDED_H, MAXPOOL2_OUT_CHANNELS)]* Ks[addr4D(ki, kj, c, f, CONV3_FILTER_HEIGHT, CONV3_FILTER_CHANNELS, CONV3_FILTER_NUMBER)];
                        }
                    }
                    output[addr3D(i, j, f, MAXPOOL3_IN_HEIGHT, CONV3_FILTER_NUMBER)] += conv_sum;
                }
            }
        }
    }

    // Étape 4 : Activation ReLU
    ly_relu: for (f = 0; f < CONV3_FILTER_NUMBER; ++f) {
        lx_relu: for (i = 0; i < MAXPOOL2_OUT_HEIGHT; ++i) {
            lz_relu: for (j = 0; j < MAXPOOL2_OUT_WIDTH; ++j) {
                output[addr3D(i, j, f, MAXPOOL3_IN_HEIGHT, CONV3_FILTER_NUMBER)] = ReLu(output[addr3D(i, j, f, MAXPOOL3_IN_HEIGHT, CONV3_FILTER_NUMBER)]);
            }
        }
    }
}

void maxpool1(
    FixedPoint input[MAXPOOL1_IN_SIZE],
    FixedPoint output[MAXPOOL1_OUT_SIZE]
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
                FixedPoint max_val = -1; // <-------- plus petit possible
                
                if (start_h + 3 > MAXPOOL1_IN_HEIGHT) {
                    if (start_w + 3 > MAXPOOL1_IN_WIDTH) {
                        // region = M[start_h:y, start_w:x, c]
                        for (i = start_h; i < MAXPOOL1_IN_HEIGHT; i++) {
                            for (j = start_w; j < MAXPOOL1_IN_WIDTH; j++) {
                                max_val = (input[addr3D(i, j, c, MAXPOOL1_IN_HEIGHT, MAXPOOL1_IN_CHANNELS)] > max_val) ? input[addr3D(i, j, c, MAXPOOL1_IN_HEIGHT, MAXPOOL1_IN_CHANNELS)] : max_val;
                            }
                        }
                    }
                    else {
                        // region = M[start_h:y, start_w:start_w+3, c]
                        for (i = start_h; i < MAXPOOL1_IN_HEIGHT; i++) {
                            for (j = start_w; j < start_w + 3; j++) {
                                max_val = (input[addr3D(i, j, c, MAXPOOL1_IN_HEIGHT, MAXPOOL1_IN_CHANNELS)] > max_val) ? input[addr3D(i, j, c, MAXPOOL1_IN_HEIGHT, MAXPOOL1_IN_CHANNELS)] : max_val;
                            }
                        }
                    }
                }
                else if (start_w + 3 > MAXPOOL1_IN_WIDTH) {
                    // region = M[start_h:start_h+3, start_w:x, c]
                    for (i = start_h; i < start_h + 3; i++) {
                        for (j = start_w; j < MAXPOOL1_IN_WIDTH; j++) {
                            max_val = (input[addr3D(i, j, c, MAXPOOL1_IN_HEIGHT, MAXPOOL1_IN_CHANNELS)] > max_val) ? input[addr3D(i, j, c, MAXPOOL1_IN_HEIGHT, MAXPOOL1_IN_CHANNELS)] : max_val;
                        }
                    }
                }
                else {
                    // region = M[start_h:start_h+3, start_w:start_w+3, c]
                    for (i = start_h; i < start_h + 3; i++) {
                        for (j = start_w; j < start_w + 3; j++) {
                            max_val = (input[addr3D(i, j, c, MAXPOOL1_IN_HEIGHT, MAXPOOL1_IN_CHANNELS)] > max_val) ? input[addr3D(i, j, c, MAXPOOL1_IN_HEIGHT, MAXPOOL1_IN_CHANNELS)] : max_val;
                        }
                    }
                }
                output[addr3D(h, w, c, MAXPOOL1_OUT_HEIGHT, MAXPOOL1_OUT_CHANNELS)] = max_val;
            }
        }
    }
}

void maxpool2(
    FixedPoint input[MAXPOOL2_IN_SIZE],
    FixedPoint output[MAXPOOL2_OUT_SIZE]
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
                FixedPoint max_val = -1; // <-------- plus petit possible
                
                if (start_h + 3 > MAXPOOL2_IN_HEIGHT) {
                    if (start_w + 3 > MAXPOOL2_IN_WIDTH) {
                        // region = M[start_h:y, start_w:x, c]
                        for (i = start_h; i < MAXPOOL2_IN_HEIGHT; i++) {
                            for (j = start_w; j < MAXPOOL2_IN_WIDTH; j++) {
                                max_val = (input[addr3D(i, j, c, MAXPOOL2_IN_HEIGHT, MAXPOOL2_IN_CHANNELS)] > max_val) ? input[addr3D(i, j, c, MAXPOOL2_IN_HEIGHT, MAXPOOL2_IN_CHANNELS)] : max_val;
                            }
                        }
                    }
                    else {
                        // region = M[start_h:y, start_w:start_w+3, c]
                        for (i = start_h; i < MAXPOOL2_IN_HEIGHT; i++) {
                            for (j = start_w; j < start_w + 3; j++) {
                                max_val = (input[addr3D(i, j, c, MAXPOOL2_IN_HEIGHT, MAXPOOL2_IN_CHANNELS)] > max_val) ? input[addr3D(i, j, c, MAXPOOL2_IN_HEIGHT, MAXPOOL2_IN_CHANNELS)] : max_val;
                            }
                        }
                    }
                }
                else if (start_w + 3 > MAXPOOL2_IN_WIDTH) {
                    // region = M[start_h:start_h+3, start_w:x, c]
                    for (i = start_h; i < start_h + 3; i++) {
                        for (j = start_w; j < MAXPOOL2_IN_WIDTH; j++) {
                            max_val = (input[addr3D(i, j, c, MAXPOOL2_IN_HEIGHT, MAXPOOL2_IN_CHANNELS)] > max_val) ? input[addr3D(i, j, c, MAXPOOL2_IN_HEIGHT, MAXPOOL2_IN_CHANNELS)] : max_val;
                        }
                    }
                }
                else {
                    // region = M[start_h:start_h+3, start_w:start_w+3, c]
                    for (i = start_h; i < start_h + 3; i++) {
                        for (j = start_w; j < start_w + 3; j++) {
                            max_val = (input[addr3D(i, j, c, MAXPOOL2_IN_HEIGHT, MAXPOOL2_IN_CHANNELS)] > max_val) ? input[addr3D(i, j, c, MAXPOOL2_IN_HEIGHT, MAXPOOL2_IN_CHANNELS)] : max_val;
                        }
                    }
                }
                output[addr3D(h, w, c, MAXPOOL2_OUT_HEIGHT, MAXPOOL2_OUT_CHANNELS)] = max_val;
            }
        }
    }
}

void maxpool3(
    FixedPoint input[MAXPOOL3_IN_SIZE],
    FixedPoint output[MAXPOOL3_OUT_SIZE]
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
                FixedPoint max_val = -1; // <-------- plus petit possible
                
                if (start_h + 3 > MAXPOOL3_IN_HEIGHT) {
                    if (start_w + 3 > MAXPOOL3_IN_WIDTH) {
                        // region = M[start_h:y, start_w:x, c]
                        for (i = start_h; i < MAXPOOL3_IN_HEIGHT; i++) {
                            for (j = start_w; j < MAXPOOL3_IN_WIDTH; j++) {
                                max_val = (input[addr3D(i, j, c, MAXPOOL3_IN_HEIGHT, MAXPOOL3_IN_CHANNELS)] > max_val) ? input[addr3D(i, j, c, MAXPOOL3_IN_HEIGHT, MAXPOOL3_IN_CHANNELS)] : max_val;
                            }
                        }
                    }
                    else {
                        // region = M[start_h:y, start_w:start_w+3, c]
                        for (i = start_h; i < MAXPOOL3_IN_HEIGHT; i++) {
                            for (j = start_w; j < start_w + 3; j++) {
                                max_val = (input[addr3D(i, j, c, MAXPOOL3_IN_HEIGHT, MAXPOOL3_IN_CHANNELS)] > max_val) ? input[addr3D(i, j, c, MAXPOOL3_IN_HEIGHT, MAXPOOL3_IN_CHANNELS)] : max_val;
                            }
                        }
                    }
                }
                else if (start_w + 3 > MAXPOOL3_IN_WIDTH) {
                    // region = M[start_h:start_h+3, start_w:x, c]
                    for (i = start_h; i < start_h + 3; i++) {
                        for (j = start_w; j < MAXPOOL3_IN_WIDTH; j++) {
                            max_val = (input[addr3D(i, j, c, MAXPOOL3_IN_HEIGHT, MAXPOOL3_IN_CHANNELS)] > max_val) ? input[addr3D(i, j, c, MAXPOOL3_IN_HEIGHT, MAXPOOL3_IN_CHANNELS)] : max_val;
                        }
                    }
                }
                else {
                    // region = M[start_h:start_h+3, start_w:start_w+3, c]
                    for (i = start_h; i < start_h + 3; i++) {
                        for (j = start_w; j < start_w + 3; j++) {
                            max_val = (input[addr3D(i, j, c, MAXPOOL3_IN_HEIGHT, MAXPOOL3_IN_CHANNELS)] > max_val) ? input[addr3D(i, j, c, MAXPOOL3_IN_HEIGHT, MAXPOOL3_IN_CHANNELS)] : max_val;
                        }
                    }
                }
                output[addr3D(h, w, c, MAXPOOL3_OUT_HEIGHT, MAXPOOL3_OUT_CHANNELS)] = max_val;
            }
        }
    }
}