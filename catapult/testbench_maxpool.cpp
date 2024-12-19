#include "functions.hpp"
#include <math.h>
#include <cmath>

#include <iostream>
#include <cstdint>
#include "ac_fixed.h"
#include "ac_math/ac_relu.h"
#include "coefs.h"
#include "cifar10_normalized_data.h"

#ifndef CCS_MAIN
#define CCS_MAIN int main
#define CCS_DESIGN(d) d
#define CCS_RETURN(a)  return a
#endif

CCS_MAIN(int argc, char **argv) {
    ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> output_conv1[24][24][64] = {0};
    ac_fixed<32 , 6, true, AC_RND_INF, AC_SAT> output_max1[12][12][64] = {0};

    convolution1(images_normalized[0], conv1_weights, conv1_biases, output_conv1);
    maxpool1(output_conv1, output_max1);

    CCS_RETURN(0);
}