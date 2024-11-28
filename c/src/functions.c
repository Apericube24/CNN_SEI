#include "functions.h"
#include <math.h>

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
