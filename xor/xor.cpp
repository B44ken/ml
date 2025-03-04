#include "boilerplate.cpp"
#include <cmath>

// model_data init_model() {
// return vector<layer_data>(2, { vec(2, 0.f), 0.f });
// }

float xor_f(float a, float b) { return (int)a ^ (int)b; }


float inference(model_data model, vec X) {
    for(int i = 0; i < model.size(); i++) {
        vec hidden = linear_multi(X, model[i].W, model[i].b);
        vec active = tanh(hidden);
        X = active;
    }
    return argmax(X);
}

int main(int argc, char** argv) {
    vec x = {0., 1.};
    if(argc == 3)
        x = {(float)atof(argv[1]), (float)atof(argv[2])};
    model_data xor_nn = {
        // layer 1
        { { {  3.789,  3.762 },
            { -0.637, -0.635 } }, { 0., 0. }
        },
        // layer 2
        { { { -6.621, -9.591 },
            {  6.621,  9.523 } }, { 0., 0. }
        }
    };
    cout << inference(xor_nn, x) << "\n";
}
