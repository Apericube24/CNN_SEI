#include "functions.hpp"
#include "ac_fixed.h"
#include "flat_image_data.h"
#include "flat_coefs.h"

#ifndef CCS_MAIN
#define CCS_MAIN int main
#define CCS_DESIGN(d) d
#define CCS_RETURN(a)  return a
#endif

void calculate3DIndices(int index, int l_x, int l_y, int l_z) {
    // Calcul des indices x, y, z à partir de l'index 1D
    int x = index / (l_y * l_z);                    // Trouver x
    int y = (index % (l_y * l_z)) / l_z;            // Trouver y
    int z = index % l_z;                            // Trouver z

    // Afficher les résultats
    std::cout << " " << index;
    std::cout << " " << x << " " << y << " " << z << " ";
}

CCS_MAIN(int argc, char **argv) {
    FixedPoint output_conv1[24*24*64] = {0};
    FixedPoint output_max1[12*12*64] = {0.000077777};
    FixedPoint padded_image1[26*26*3] = {0};

    FixedPoint output_conv2[12*12*32] = {0};
    FixedPoint output_max2[6*6*32] = {0};
    FixedPoint padded_image2[14*14*64] = {0};

    FixedPoint output_conv3[6*6*20] = {0};
    FixedPoint output_max3[3*3*20] = {0};
    FixedPoint padded_image3[8*8*32] = {0};

    FixedPoint output_reshape[180] = {0};
    FixedPoint output_fcp[10] = {0};

    convolution1(image_1, conv1_weights, conv1_biases, output_conv1, padded_image1);
    maxpool1(output_conv1, output_max1);

    convolution2(output_max1, conv2_weights, conv2_biases, output_conv2, padded_image2);
    maxpool2(output_conv2, output_max2);

    convolution3(output_max2, conv3_weights, conv3_biases, output_conv3, padded_image3);
    maxpool3(output_conv3, output_max3);

    reshape(output_max3, output_reshape);
    FCP(output_reshape, local3_weights, local3_biases, output_fcp);

    int i = 0;

    for (i = 0; i < 10; i++) {
        // std::cout << i << std::endl;
        // calculate3DIndices(i, 1, 1, 10);
        std::cout << output_fcp[i] << std::endl;
    }

    CCS_RETURN(0);
}