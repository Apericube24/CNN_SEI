#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

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


ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> ReLu(ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> x);

void convolution1(
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> image[IMG_SIZE],
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> Ks[CONV1_SIZE],
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> biais[CONV1_BIAS_NUMBER],
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> output[MAXPOOL1_IN_SIZE]
);

#endif