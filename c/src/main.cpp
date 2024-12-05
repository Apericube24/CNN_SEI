#include "functions.hpp"
#include "coefs.h"
#include "cifar10_data.h"
#include "cifar10_normalized_data.h"
#include <iostream>
#include <cmath>


int main() {
    // double normalized[24][24][3] = {0};

    double output_conv1[24][24][64] = {0};
    double output_max1[12][12][64] = {0};

    double output_conv2[12][12][32] = {0};
    double output_max2[6][6][32] = {0};

    double output_conv3[6][6][20] = {0};
    double output_max3[3][3][20] = {0};

    double output_reshape[180] = {0};
    double output_fcp[10] = {0};

    // normalize_image_old<24, 24, 3>(images[0], normalized);

    convolution_old<24, 24, 3, 3, 3, 64>(images_normalized[0], conv1_weights, conv1_biases, output_conv1);
    maxpool_3x3<24, 24, 64, 12, 12>(output_conv1, output_max1);

    convolution_old<12, 12, 64, 3, 3, 32>(output_max1, conv2_weights, conv2_biases, output_conv2);
    maxpool_3x3<12, 12, 32, 6, 6>(output_conv2, output_max2);

    convolution_old<6, 6, 32, 3, 3, 20>(output_max2, conv3_weights, conv3_biases, output_conv3);
    maxpool_3x3<6, 6, 20, 3, 3>(output_conv3, output_max3);

    reshape_old(output_max3, output_reshape);
    FCP_old(output_reshape, local3_weights, local3_biases, output_fcp);
    int i = 0;

    for (i = 0; i < 10; i++) {
        printf("%.10lf\n", output_fcp[i]);
    }
}