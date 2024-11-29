#include "functions.hpp"
#include <iostream>

#define IN_HEIGHT 6
#define IN_WIDTH 6
#define NUM_CHANNELS 1
#define OUT_HEIGHT 3
#define OUT_WIDTH 3

int main() {

    float input[IN_HEIGHT][IN_WIDTH][NUM_CHANNELS] = {
        {{1}, {2}, {3}, {4}, {5}, {6}},
        {{7}, {8}, {9}, {10}, {11}, {12}},
        {{13}, {14}, {15}, {16}, {17}, {18}},
        {{19}, {20}, {21}, {22}, {23}, {24}},
        {{25}, {26}, {27}, {28}, {29}, {30}},
        {{31}, {32}, {33}, {34}, {35}, {36}}
    };
    float output[OUT_HEIGHT][OUT_WIDTH][NUM_CHANNELS] = {0};


    maxpool_3x3<IN_HEIGHT, IN_WIDTH, NUM_CHANNELS, OUT_HEIGHT, OUT_WIDTH>(input, output);

    return 0;
}