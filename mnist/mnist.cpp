#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include "json.h"
using namespace std;
using json = nlohmann::json;

typedef struct { vector<vector<float>> coef; vector<float> inter; } model_data;
typedef struct { vector<float> X; int y; } sample_data;

model_data init_model_data(size_t out_dim, size_t in_dim) {
    model_data m;
    m.coef.resize(out_dim, vector<float>(in_dim, 0.0f));
    m.inter.resize(out_dim, 0.0f);
    return m;
}

model_data load_model(string path) {
    json content = json::parse(ifstream(path.c_str()));
    return {
        content["coef"].get<vector<vector<float>>>(),
        content["inter"].get<vector<float>>()
    };
}

sample_data load_sample(string path) {
    ifstream f(path.c_str());
    if (!f.is_open()) {
        cout << "Error: file not found - " << path << "\n";
        exit(1);
    }
    json content = json::parse(f);
    return {
        content["X"].get<vector<float>>(),
        content["y"].get<int>()
    };
}

int argmax(vector<float> a) {
    int best = 0;
    for(int i = 0; i < a.size(); i++)
        if(a[i] > a[best]) best = i;
    return best;
}

float dot(vector<float> a, vector<float> b) { return inner_product(a.begin(), a.end(), b.begin(), 0.f); }

vector<float> scale(float x, vector<float> v) {
    for(int i = 0; i < v.size(); i++)
        v[i] *= x;
    return v;
}

vector<float> sum(vector<float> a, vector<float> b) {
    for(int i = 0; i < a.size(); i++)
        a[i] *= b[i];
    return a;
}

vector<float> logit(vector<vector<float>> A, vector<float> X, vector<float> b) {
    auto all = vector<float>(A.size(), 0);
    for(int i = 0; i < A.size(); i++)
        all[i] = dot(A[i], X) + b[i];
    return all;
}

vector<float> softmax(vector<float> a) {
    float denom = 0;
    for(int i = 0; i < a.size(); i++)
        denom += exp(a[i]);

    auto softed = vector<float>(a.size(), 0);
    for(int i = 0; i < a.size(); i++)
        softed[i] = exp(a[i]) / denom;

    return softed;
}

vector<float> infer(model_data model, sample_data sample) {
    return softmax(logit(model.coef, sample.X, model.inter));
}

float dlt(int a, int b) { return (a == b) ? 1 : 0; }

model_data gradients(model_data model, sample_data sample, vector<float> pred) {
    auto grad = init_model_data(model.coef.size(), model.coef[0].size());

    for(int i = 0; i < model.coef.size(); i++) {
        auto dL_dpi = pred[i] - dlt(i, sample.y);
        auto dL_dAi = scale(dL_dpi, sample.X);
        auto dL_dbi = dL_dpi;

        grad.coef[i] = dL_dAi;
        grad.inter[i] += dL_dbi;
    }

    return grad;
}

const float lr = 0.001;
model_data apply_grads(model_data model, model_data grad) {
    for (int i = 0; i < model.coef.size(); i++) {
        for (int j = 0; j < model.coef[i].size(); j++)
            model.coef[i][j] -= lr * grad.coef[i][j];
        model.inter[i] -= lr * grad.inter[i];
    }
    return model;
}

model_data train_one(model_data model, sample_data sample) {
    auto current_pred = infer(model, sample);
    auto grad = gradients(model, sample, current_pred);
    model_data update = apply_grads(model, grad);
    return update;
}

int accuracy(model_data model, vector<sample_data> tests) {
    int good = 0;
    for(int i = 0; i < tests.size(); i++)
        good += (argmax(infer(model, tests[i])) == tests[i].y);
    return good;
}

int main(int argc, char** argv) {
    int total_tests = 100, total_trains = 600, i_test = 0, i_train = 0;
    vector<sample_data> train_samples, test_samples;

    for(int i = 1; i <= total_trains; i++)
        train_samples.push_back(load_sample("dataset/train" + to_string(i) + ".json"));
    for (int i = 1; i <= total_tests; i++)
        test_samples.push_back(load_sample("dataset/test" + to_string(i) + ".json"));

    auto model = init_model_data(10, 28*28);
    for(int epoch = 0; epoch <= 100; epoch++) {
        if(epoch % 10 == 0)
            cout << "epoch = " << epoch << " \tacc = " << accuracy(model, test_samples) << "%\n";
        for(int i = 0; i < total_trains; i++) {
            i_test = (i_test + 1) % total_trains;
            model = train_one(model, train_samples[i]);
        }
    }
}
