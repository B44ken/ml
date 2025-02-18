#include "shittyml.h"
#include <vector>

namespace shittyml {
    using namespace std;

    Layer::Layer() = default;

    NoOpLayer::NoOpLayer() = default;

    vec NoOpLayer::forward(vec input) {
        return input;
    }

    int test = 7;

    Model::Model(initializer_list<Layer*> layers) : pipeline(layers) {};

    vec Model::forward(vec input) {
        vec out = input;
        for (auto layer : pipeline)
            out = layer->forward(out);
        return out;
    }
};