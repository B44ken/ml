#include "shittyml.h"
using namespace shittyml;

Trainer::Trainer(vec2d X, vec y_true) : X(X), y_true(y_true) {};

float Trainer::mse(Model m) {
    return vec(y_true).map([this, &m](float y, int i) {
        float err = m.forward(X[i])[0] - y;
        return err * err;
        }).sum();
}

vec grad_square_errors(vec t, vec p) {
    return vec(t).map([&t, &p](float y, int i) {
        return 2 * (p[i] - t[i]);
        });
}

void Trainer::train_epochs(Model* m, int n) {
    for (int i = 0; i < n; i++)
        backprop(m);
}

int Trainer::train_until_convergence(Model* m) {
    float threshold = 0.00001;
    float l = mse(*m);
    for (int i = 0; i < 1000 * 1000; i++) {
        backprop(m);
        if (i % 20 != 0)
            continue;
        float l2 = mse(*m);
        if (l - l2 < threshold) return i;
        l = l2;
    }
    return -1;
}

void single_layer_backprop_one(Model* model, vec Xi, float yi) {
    auto pl = &model->pipeline;

    // forward pass
    auto forward = vec2d(pl->size(), 1);
    for (size_t i = 0; i < pl->size(); i++) {
        auto inp = (i == 0) ? Xi : forward[i - 1];
        forward[i] = (*pl)[i]->forward(inp);
    }

    // backward pass
    auto grad_back = grad_square_errors(vec({ yi }), forward.back());

    for (int i = pl->size() - 1; i >= 0; i--) {
        auto linear = dynamic_cast<Linear*>((*pl)[i]);
        if (linear == nullptr) continue;

        auto inp = (i == 0) ? Xi : forward[i - 1];
        auto g = linear->grad(inp, grad_back);
        linear->apply_grad(g);
    }
}

void Trainer::backprop_one(Model* model, vec Xi, float yi) {
    auto pl = &model->pipeline;

    // forward pass
    auto forward = vec2d(pl->size(), 1);
    for (size_t i = 0; i < pl->size(); i++) {
        auto inp = (i == 0) ? Xi : forward[i - 1];
        forward[i] = (*pl)[i]->forward(inp);
    }

    // backward pass
    auto grad_back = grad_square_errors(vec({ yi }), forward.back());

    for (int i = pl->size() - 1; i >= 0; i--) {
        auto linear = dynamic_cast<Linear*>((*pl)[i]);
        if (linear == nullptr) continue;

        auto inp = (i == 0) ? Xi : forward[i - 1];
        auto g = linear->grad(inp, grad_back);
        linear->apply_grad(g);

        if(i == 0) continue;;

        grad_back = linear->backward_grad(grad_back);
    }
}


void Trainer::backprop(Model* model) {
    for (size_t i = 0; i < X.size(); i++)
        backprop_one(model, X[i], y_true[i]);
}