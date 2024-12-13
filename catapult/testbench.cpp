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
    ac_fixed<128, 6, true, AC_RND_INF, AC_SAT> output_conv1[24][24][64] = {0};
    ac_fixed<128 , 6, true, AC_RND_INF, AC_SAT> output_max1[12][12][64] = {0};

    // ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> output_conv2[12][12][32] = {0};
    // ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> output_max2[6][6][32] = {0};

    // ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> output_conv3[6][6][20] = {0};
    // ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> output_max3[3][3][20] = {0};

    // ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> output_reshape[180] = {0};
    // ac_fixed<32, 6, true, AC_RND_INF, AC_SAT> output_fcp[10] = {0};

    for (size_t i = 0; i < 64; ++i) {
        std::cout << "Element " << i << ": " << conv1_biases[i] << std::endl;
    }

    convolution1(images_normalized[0], conv1_weights, conv1_biases, output_conv1);
    // maxpool1(output_conv1, output_max1);

    // convolution2(output_max1, conv2_weights, conv2_biases, output_conv2);
    // maxpool2(output_conv2, output_max2);

    // convolution3(output_max2, conv3_weights, conv3_biases, output_conv3);
    // maxpool3(output_conv3, output_max3);

    // reshape(output_max3, output_reshape);
    // FCP(output_reshape, local3_weights, local3_biases, output_fcp);
}