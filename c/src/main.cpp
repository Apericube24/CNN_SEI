#include <array>
#include "virgule_fixe.h"
#include "functions.hpp"

using fixed_t = FixedPoint<int32_t, 16>;

constexpr size_t IN_H = 24, IN_W = 24, IN_C = 64;
constexpr size_t OUT_H = 12, OUT_W = 12;

constexpr size_t IMG_H = 24, IMG_W = 24, IMG_C = 3, K_H = 3, K_W = 3, NUM_FILTERS = 64;

int main() {

    std::array<std::array<std::array<fixed_t, IN_C>, IN_W>, IN_H> input = {0};
    std::array<std::array<std::array<fixed_t, IN_C>, OUT_W>, OUT_H> output = {0};

    maxpool_1<IN_H, IN_W, IN_C, OUT_H, OUT_W>(input, output);

    std::array<std::array<std::array<float, IMG_C>, IMG_W>, IMG_H> image = {0}; 
    std::array<std::array<std::array<std::array<float, NUM_FILTERS>, IMG_C>, K_W>, K_H> Ks = {0};
    std::array<float, NUM_FILTERS> biais = {0}; 
    std::array<std::array<std::array<float, NUM_FILTERS>, IMG_W>, IMG_H> output = {0}; 

    convolution<IMG_H, IMG_W, IMG_C, K_H, K_W, NUM_FILTERS>(image, Ks, biais, output);
}