#include "onef.h"
using namespace onef;

// relu

ReLU::ReLU() = default;

void ReLU::apply_grad(LinearGrad grad) {}

vec ReLU::forward(vec a) {
    last_input = a;
    return a.map([](float x) { return x > 0 ? x : 0; });
}

LinearGrad ReLU::grad(vec input, vec error) {
    auto g = LinearGrad(new Linear(vec2d(input.size(), 1), vec(input.size())));
    for (size_t i = 0; i < input.size(); i++)
        g.W[i][0] = input[i] > 0 ? error[i] : 0;
    return g;
}

vec ReLU::backward_grad(vec error) {

    auto output = vec(last_input.size());
    for(size_t i = 0; i < last_input.size(); i++)
        output[i] = last_input[i] > 0 ? error[i] : 0;

    return output;
}

// sigmoid

Sigmoid::Sigmoid() = default;

void Sigmoid::apply_grad(LinearGrad grad) {}

vec Sigmoid::forward(vec a) {
    last_input = a;
    return a.map([](float x) { return 1 / (1 + exp(-x)); });
}

LinearGrad Sigmoid::grad(vec input, vec error) {
    auto g = LinearGrad(new Linear(vec2d(input.size(), 1), vec(input.size())));
    for (size_t i = 0; i < input.size(); i++) {
        float s = 1 / (1 + exp(-input[i]));
        g.W[i][0] = error[i] * s * (1 - s);
    }
    return g;
}

vec Sigmoid::backward_grad(vec error) {
    return vec(error.size()).map([&](float _, int i) {
        float s = 1 / (1 + exp(-last_input[i]));
        return error[i] * s * (1 - s);
    });
}