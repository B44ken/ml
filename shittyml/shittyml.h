#pragma once

#include <vector>
#include <iostream>
#include <functional>
#include <cmath>

namespace shittyml {
    using namespace std;

    class Linear;
    class LinearGrad;

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

    class vec2d : public vector<vec> {
    public:
        vec2d();
        vec2d(int c, int r);
        vec2d(initializer_list<initializer_list<float>> rows);
        vec2d transpose();
        friend ostream& operator<<(ostream& os, vec2d v);
        vec operator*(vec a);
    };

    class Layer {
    public:
        vec shape;
        Layer() = default;
        ~Layer() = default;
        virtual vec forward(vec input) = 0;
        virtual LinearGrad grad(vec input, vec backflow) = 0;
        virtual void apply_grad(LinearGrad grad) = 0;
        virtual vec backward_grad(vec error);
    };

    class LinearGrad {
    public:
        vec2d W;
        vec b;
        LinearGrad(Linear *shape);
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
        vec backward_grad(vec error);
        void apply_grad(LinearGrad grad);
        LinearGrad grad(vec fwd_input, vec grad_back_flow);
    };
    

    class ReLU : public Layer {
    public:
        ReLU();
        void apply_grad(LinearGrad grad);
        vec forward(vec input);
        LinearGrad grad(vec _1, vec _2);
        vec backward_grad(vec error);
    };

    class Sigmoid : public Layer {
    public:
        Sigmoid();
        void apply_grad(LinearGrad grad);
        vec forward(vec input);
        LinearGrad grad(vec _1, vec _2);
        vec backward_grad(vec error);
    };

    class Trainer {
    public:
        vec2d X;
        vec y_true;

        Trainer(vec2d X, vec y_true);

        float mse(Model m);
        vec grad(Linear l);
        void train_epochs(Model* m, int epochs);
        int train_until_convergence(Model* m);
        void backprop_one(Model* model, vec Xi, float y_truei);
        void backprop(Model* model);
    };
}