#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

#include <iostream>

float ReLu(float x);
void softmax(float x[10]);
void reshape(float input[3][3][20], float output[180]);
void FCP(float M[180], float weights[180][10], float bias[10], float output[10]);

template <int HEIGHT, int WIDTH, int NUM_CHANNELS, int OUT_HEIGHT, int OUT_WIDTH>
void maxpool_3x3(float input[HEIGHT][WIDTH][NUM_CHANNELS], float output[OUT_HEIGHT][OUT_WIDTH][NUM_CHANNELS]) {
    int c = 0;
    int h = 0;
    int w = 0;
    int start_h, start_w;
    for (c = 0; c < NUM_CHANNELS; c++) {
        for (h = 0; h < OUT_HEIGHT; h++) {
            for (w = 0; w < OUT_WIDTH; w++) {
                start_h = h * 2;
                start_w = w * 2;
                float max_val = -1; // <-------- plus petit possible
                
                if (start_h + 3 > HEIGHT) {
                    if (start_w + 3 > WIDTH) {
                        // region = M[start_h:y, start_w:x, c]
                        for (int i = start_h; i < HEIGHT; i++) {
                            for (int j = start_w; j < WIDTH; j++) {
                                max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
                            }
                        }
                    }
                    else {
                        // region = M[start_h:y, start_w:start_w+3, c]
                        for (int i = start_h; i < HEIGHT; i++) {
                            for (int j = start_w; j < start_w + 3; j++) {
                                max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
                            }
                        }
                    }
                }
                else if (start_w + 3 > WIDTH) {
                    // region = M[start_h:start_h+3, start_w:x, c]
                    for (int i = start_h; i < start_h + 3; i++) {
                        for (int j = start_w; j < WIDTH; j++) {
                            max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
                        }
                    }
                }
                else {
                    // region = M[start_h:start_h+3, start_w:start_w+3, c]
                    for (int i = start_h; i < start_h + 3; i++) {
                        for (int j = start_w; j < start_w + 3; j++) {
                            max_val = (input[i][j][c] > max_val) ? input[i][j][c] : max_val;
                        }
                    }
                }
                output[h][w][c] = max_val;
            }
        }
    }
}

template <int IMG_H, int IMG_W, int IMG_C, int K_H, int K_W, int NUM_FILTERS>
void convolution(const float image[IMG_H][IMG_W][IMG_C],
                 const float Ks[K_H][K_W][IMG_C][NUM_FILTERS],
                 const float biais[NUM_FILTERS],
                 float output[IMG_H][IMG_W][NUM_FILTERS]) {

    int PADDING = 1;
    int PADDED_H = IMG_H + 2 * PADDING;
    int PADDED_W = IMG_W + 2 * PADDING;

    float padded_image[PADDED_H][PADDED_W][IMG_C];
    // padding
    for (int c = 0; c < IMG_C; ++c) {
        for (int i = 0; i < IMG_H + 2 * PADDING; ++i) {
            for (int j = 0; j < IMG_W + 2 * PADDING; ++j) {
                if (i == 0 || j == 0 || i == IMG_H + PADDING || j == IMG_W + PADDING) {
                    padded_image[i][j][c] = 0.0;
                }
                else {
                    padded_image[i][j][c] = image[i - 1][j - 1][c];
                }
            }
        }
    }
    std::cout << "padded_image:\n";
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            std::cout << padded_image[i][j][0] << " "; // Only one filter
        }
        std::cout << "\n";
    }

    // init biais
    for (int f = 0; f < NUM_FILTERS; ++f) {
        for (int i = 0; i < IMG_H; ++i) {
            for (int j = 0; j < IMG_W; ++j) {
                output[i][j][f] = biais[f];
            }
        }
    }

    // etape conv
    for (int f = 0; f < NUM_FILTERS; ++f) {
        for (int c = 0; c < IMG_C; ++c) {
            for (int i = 0; i < IMG_H; ++i) {
                for (int j = 0; j < IMG_W; ++j) {
                    float conv_sum = 0;
                    for (int ki = 0; ki < K_H; ++ki) {
                        for (int kj = 0; kj < K_W; ++kj) {
                            conv_sum = conv_sum + padded_image[i + ki][j + kj][c] * Ks[ki][kj][c][f];
                        }
                    }
                    output[i][j][f] = output[i][j][f] + conv_sum;
                }
            }
        }

        // etape relu
        for (int i = 0; i < IMG_H; ++i) {
            for (int j = 0; j < IMG_W; ++j) {
                output[i][j][f] = ReLu(output[i][j][f]);
            }
        }
    }
}

template <int IMG_H, int IMG_W, int IMG_C>
void normalize_image(float image[IMG_H][IMG_W][IMG_C], float normalized[IMG_H][IMG_W][IMG_C]) {
    const int N = IMG_H * IMG_W * IMG_C;

    // Step 1: Calculate mean
    float total = 0.0f;
    for (int i = 0; i < IMG_H; ++i) {
        for (int j = 0; j < IMG_W; ++j) {
            for (int k = 0; k < IMG_C; ++k) {
                total += image[i][j][k];
            }
        }
    }
    float mean_value = total / N;

    // Step 2: Calculate variance
    float variance = 0.0f;
    for (int i = 0; i < IMG_H; ++i) {
        for (int j = 0; j < IMG_W; ++j) {
            for (int k = 0; k < IMG_C; ++k) {
                float diff = image[i][j][k] - mean_value;
                variance += diff * diff;
            }
        }
    }
    float sigma = std::sqrt(variance / N);

    // Print mean and sigma
    std::cout << "Mean: " << mean_value << " ------- Sigma: " << sigma << std::endl;

    // Step 3: Normalize the image
    const float epsilon = 1.0f / std::sqrt(N); // Small value to avoid division by zero
    float denominator = std::max(sigma, epsilon);

    for (int i = 0; i < IMG_H; ++i) {
        for (int j = 0; j < IMG_W; ++j) {
            for (int k = 0; k < IMG_C; ++k) {
                normalized[i][j][k] = (image[i][j][k] - mean_value) / denominator;
            }
        }
    }
}

#endif
