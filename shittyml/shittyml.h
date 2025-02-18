#pragma once

#include <vector>
#include <iostream>

namespace shittyml {
    using namespace std;

    // typedef vector<float> vec;
    class vec : public vector<float> {
        public:
        vec();
        vec(initializer_list<float> list);
        vec(int size);
        string stringify() const;
        friend ostream &operator<<(ostream &os, vec v);
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
        Linear(vec2d weights, vec bias);;
        vec forward(vec input);
    };

    class ReLU : public Layer {
        public:
        vec forward(vec input);
    };
}