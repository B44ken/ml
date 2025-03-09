// this is probably broken but ok for reference ig

#include "boilerplate.h"

float inference(model_data model, vec X) {
    for(int i = 0; i < model.size(); i++) {
        if(model[i].tag == "do_relu")
            X = relu(X);
        else
            X = logit(X, model[i]);
    }
    X = logit(X, model[model.size()-1]);
    return X[0] > 0.5;
}

int main(int argc, char** argv) {
    vec x = (argc == 3) ? vec({(float)atof(argv[1]), (float)atof(argv[2])}) : vec({0, 1});
    model_data xor_nn = {
        {{{1, -1}, {-1, 1}}, {0, 0}},
        do_relu,
        {{{1, 1}}, {0, 0}}
    };
    cout << inference(xor_nn, x) << "\n";
}
