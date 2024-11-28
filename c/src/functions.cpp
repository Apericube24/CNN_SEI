#include "functions.hpp"
#include <math.h>

// #include "virgule_fixe.hpp"
#include <iostream>
#include <cstdint> 


// using fixed_t = FixedPoint<int32_t, 16>; 

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

void reshape(float input[3][3][20], float output[180]) {
    int index = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 20; k++) {
                output[index++] = input[i][j][k];
            }
        }
    }
}

void FCP(float M[180], float weights[180][10], float bias[10], float output[10]) {
    // NON  TESTE ATTENTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    float temp_output[10] = {0};

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 180; j++) {
            temp_output[i] += M[j] * weights[j][i];
        }
        temp_output[i] += bias[i];
    }

    softmax(temp_output);

    for (int i = 0; i < 10; i++) {
        output[i] = temp_output[i];
    }
}