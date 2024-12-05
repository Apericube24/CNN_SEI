#include "functions.hpp"
#include <math.h>
#include <cmath>

#include <iostream>
#include <cstdint>
#include "ac_fixed.h"
#include "ac_math/ac_relu.h"
#include "coefs.h"
#include "cifar10_normalized_data.h"

CCS_MAIN(int argc, char **argv) {
    ac_fixed<32, 6, true, AC_RBD_INF, AC_SAT> output_conv1[24][24][64] = {0};

    convolution1(images_normalized[0], conv1_weights, conv1_biases, output_conv1);
}