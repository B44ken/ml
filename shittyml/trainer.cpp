#include "shittyml.h"
using namespace shittyml;

Trainer::Trainer(vec2d X, vec y_true) : X(X), y_true(y_true) {};

float Trainer::mean_sq_error(Model m) {
    return vec(y_true).map([this, &m](float y, int i) {
        float err = m.forward(X[i])[0] - y;
        return err * err;
    }).sum();
}

vec Trainer::grad(Linear l) {
    float m = l.W[0][0], b = l.b[0];
    for(size_t i = 0; i < y_true.size(); i++) {
        auto dl = l.grad(X[i], y_true[i]);
        m += dl.W[0][0];
        b += dl.b[0];
    }
    return vec({m, b});
}

void Trainer::apply_grad(Linear* l, vec grad) {
    float lr = .001;
    l->W[0][0] -= lr * grad[0];
    l->b[0] -= lr * grad[1];
}

void Trainer::train_epochs(Model* m, int i) {
    for(int j = 0; j < i; j++) {
        auto l = (Linear*)m->pipeline[0];
        auto grad = this->grad(*l);
        this->apply_grad(l, grad);
    }
}