#include "onef.h"

namespace onef {
    using namespace std;

    // Layer::Layer() = default;

    vec Layer::backward_grad(vec error) {

        return error;
    }

    NoOpLayer::NoOpLayer() = default;

    vec NoOpLayer::forward(vec input) { return input; }
    LinearGrad NoOpLayer::grad(vec input, vec backflow) {
        return LinearGrad(new Linear(0, 0));
    }
    void NoOpLayer::apply_grad(LinearGrad grad) {}
    vec NoOpLayer::backward_grad(vec error) {
        return error;
    }

    Model::Model(initializer_list<Layer*> layers) : pipeline(layers) {};

    vec Model::forward(vec input) {
        vec out = input;
        for(size_t i = 0; i < pipeline.size(); i++)
            out = pipeline[i]->forward(out);
        return out;
    }
};