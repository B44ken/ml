#include "shittyml.h"

using namespace shittyml;

// a Linear unit exists to forward pass y_i = aX_i + b_i for all i
// that is, we apply |i| dot products to the same input vector, plus a bias
Linear::Linear(vec2d weights, vec bias) : W(weights), b(bias) {}

// a Linear unity of `out` vectors that each take `in` inputs
Linear::Linear(int in, int out) {
    W = vec2d(in, out);
    b = vec(out);

    for(size_t Wx = 0; Wx < W.size(); Wx++) {
        for(size_t Wy = 0; Wy < W[Wx].size(); Wy++) {
            W[Wx][Wy] = 1.0;
        }
        b[Wx] = 1.0;
    }

}

// w.X + b
float dot(vec X, vec w, float b) {
    return vec(X).map([X, w](float _, int i) { return X[i] * w[i]; }).sum() + b;
}

// do w.Xi + bi for all i
vec Linear::forward(vec x) {
    last_input = x;
    return vec(W.size()).map([this, &x](float _, int i) { return dot(x, W[i], b[i]); });
}

vec Linear::backward_grad(vec error) {


    
    // Calculate W^T * error to propagate gradients backward


    vec result(W[0].size());
    
    for (size_t j = 0; j < W[0].size(); j++) {
        for (size_t i = 0; i < W.size(); i++) {
            result[j] += W[i][j] * error[i];
        }
    }
    

    return result;
}

LinearGrad Linear::grad(vec input, vec error) {
    // dL/dW = error * fwd_input
    // dL/db = error
    auto g = LinearGrad(this);
    for (size_t i = 0; i < W.size(); i++) {
        for (size_t j = 0; j < W[i].size(); j++) {
            g.W[i][j] = error[i] * input[j];
        }
        g.b[i] = error[i];
    }
    return g;
}

float lr = 0.00001;
// subtract lr*grad from each element of W and b
void Linear::apply_grad(LinearGrad grad) {
    for (size_t i = 0; i < W.size(); i++) {
        for (size_t j = 0; j < W[i].size(); j++) {
            W[i][j] -= lr * grad.W[i][j];
        }
        b[i] -= lr * grad.b[i];
    }
}

LinearGrad::LinearGrad(Linear* shape) {
    W = vec2d(shape->W.size(), shape->W[0].size());
    b = vec(shape->b.size());
}