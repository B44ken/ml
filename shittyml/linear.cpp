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

// w.X + b
float dot(vec X, vec w, float b) {
    return vec(X).map([w, b](float x, int i) { return x * w[i]; }).sum() + b;
}

// do w.Xi + bi for all i
vec Linear::forward(vec x) {
    return vec(x).map([this, &x](float _, int i) { return dot(x, W[i], b[i]); });
}

Linear Linear::grad(vec x, float y) {
    vec2d dWdx(W.size(), vec(W[0].size()));
    vec dbdx(b.size());

    for(size_t i = 0; i < W.size(); i++) {
        // chain rule: dL/dx = dL/dF * dF/dx
        // dL/dF = 2(f(x) - F(x))
        float diff = 2 * (forward(x)[i] - y);
        // dF/dx = x for now (should pass chain rule through the model)
        dWdx[i] = vec(x).map([this, &x, diff, i](float _, int j) { return diff * x[j]; });
        // dF/db = 1
        dbdx[i] = diff; 
    }

    return Linear(dWdx, dbdx);
}