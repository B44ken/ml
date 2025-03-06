#include "shittyml.h"
using namespace shittyml;

ReLU::ReLU() = default;

void ReLU::apply_grad(LinearGrad grad) {}

vec ReLU::forward(vec a) {
    return a.map([](float x) { return x > 0 ? x : 0; });
}

LinearGrad ReLU::grad(vec _1, vec _2) {
    return LinearGrad(new Linear(vec2d(1, 1), vec(_1.size())));
}

vec ReLU::backward_grad(vec error) {
    return error.map([](float x) { return x > 0 ? x : 0; });
}

Sigmoid::Sigmoid() = default;

void Sigmoid::apply_grad(LinearGrad grad) {}

vec Sigmoid::forward(vec a) {
    return a.map([](float x) { return 1 / (1 + exp(-x)); });
}

LinearGrad Sigmoid::grad(vec _1, vec _2) {
    return LinearGrad(new Linear(vec2d(_1.size(), 1), vec(_1.size())));
}

vec Sigmoid::backward_grad(vec error) {
    return error.map([](float x) { return x * (1 - x); });
}