#include "shittyml.h"
using namespace std;

class Trainer {
    public:
    shittyml::vec2d X;
    shittyml::vec y_true;

    Trainer(shittyml::vec2d X, shittyml::vec y_true) : X(X), y_true(y_true) {};

    float mean_sq_error(shittyml::Model m) {
        return shittyml::vec(y_true).map([this, &m](float y, int i) {
            float err = m.forward(X[i])[0] - y;
            return err * err;
        }).sum();
    }

    shittyml::vec grad(shittyml::Linear l) {
        float m = l.W[0][0], b = l.b[0];
        for(size_t i = 0; i < y_true.size(); i++) {
            auto dl = l.grad(X[i], y_true[i]);
            m += dl.W[0][0];
            b += dl.b[0];
        }
        return shittyml::vec({m, b});
    }

    void apply_grad(shittyml::Linear* l, shittyml::vec grad) {
        float lr = .001;
        l->W[0][0] -= lr * grad[0];
        l->b[0] -= lr * grad[1];
    }

    void train_epochs(shittyml::Model* m, int i) {
        for(int j = 0; j < i; j++) {
            auto l = (shittyml::Linear*)m->pipeline[0];
            auto grad = this->grad(*l);
            this->apply_grad(l, grad);
        }
    }
};

int main() {
    auto model = shittyml::Model({
        new shittyml::Linear(1, 1)
    });

    auto mult_two = Trainer(
        { {0.3}, {1.6}, {1.9}, {2.5} },
        { 1.1,    1.4,   2.0,   2.1  }
    );

    cout << "loss before\t" << mult_two.mean_sq_error(model) << endl;
    mult_two.train_epochs(&model, 1000000);
    cout << "loss after\t" << mult_two.mean_sq_error(model) << endl;
    float b = model.forward({0})[0];
    float m = model.forward({1})[0] - b;
    cout << "y = " << m << "x + " << b << endl;
}