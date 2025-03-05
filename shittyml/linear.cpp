#include "shittyml.h"

using namespace shittyml;

// a Linear unit exists to forward pass y_i = aX_i + b_i for all i
// that is, we apply |i| dot products to the same input vector, plus a bias
Linear::Linear(vec2d weights, vec bias) : W(weights), b(bias) {}

// a Linear unity of `out` vectors that each take `in` inputs
Linear::Linear(int in, int out) {
    W = vec2d(out, vec(in));
    b = vec(out);
}

float dot(vec X, vec w, float b) {
    return vec(X).map([w, b](float x, int i) { return x * w[i]; }).sum() + b;
}

vec Linear::forward(vec x) {
    return vec(x).map([this, &x](float _, int i) { return dot(x, W[i], b[i]); });
}

// L = (f(x) - F(x))^2
// dL/dx = 2(f(x) - F(x)) * df/dx
Linear Linear::grad(vec x, float y) {
    if(W.size() != 1 || b.size() != 1)
        throw "y = mx+b only";

    float diff = 2 * (forward(x)[0] - y);
    auto grads = Linear({{ diff * x[0] }}, { diff });

    return grads;
}