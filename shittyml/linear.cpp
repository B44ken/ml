#include "shittyml.h"

using namespace shittyml;

Linear::Linear(vec2d weights, vec bias) : W(weights), b(bias) {}

float dot_b(vec x, vec w, float b) {
    float out = 0;
    for (int i = 0; i < x.size(); i++)
        out += x[i] * w[i];
    return out + b;
}

vec Linear::forward(vec x) {
    auto out = vec(W.size());
    for (int i = 0; i < W.size(); i++)
        out[i] = dot_b(x, W[i], b[i]);
    return out;
}