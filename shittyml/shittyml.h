#pragma once

#include <vector>
#include <iostream>
#include <functional>

namespace shittyml {
    using namespace std;

    class vec : public vector<float> {
    public:
        vec();
        vec(initializer_list<float> list);
        vec(int size);
        vec map(function<float(float, int)> f);
        vec map(function<float(float)> f);
        float sum();
        string stringify() const;
        friend ostream& operator<<(ostream& os, vec v);
        vec operator*(float c);
        vec operator+(float b);
    };

    typedef vector<vec> vec2d;

    class Layer {
    public:
        vec shape;
        Layer();
        virtual vec forward(vec input) = 0;
        virtual ~Layer() = default;
    };

    class NoOpLayer : public Layer {
    public:
        NoOpLayer();
        vec forward(vec input);
    };

    class Model {
    public:
        vector<Layer*> pipeline;
        Model(initializer_list<Layer*> layers);
        vec forward(vec input);
    };

    class Linear : public Layer {
    public:
        vec2d W;
        vec b;
        Linear(vec2d weights, vec bias);
        Linear(int in, int out);
        vec forward(vec input);
        Linear grad(vec X, float y);
    };

    class ReLU : public Layer {
    public:
        vec forward(vec input);
    };

    class Trainer {
    public:
        shittyml::vec2d X;
        shittyml::vec y_true;

        Trainer(shittyml::vec2d X, shittyml::vec y_true);

        float mean_sq_error(shittyml::Model m);
        shittyml::vec grad(shittyml::Linear l);
        void apply_grad(shittyml::Linear* l, shittyml::vec grad);
        void train_epochs(shittyml::Model* m, int epochs);
    };
}