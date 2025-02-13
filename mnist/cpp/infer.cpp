#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include "json.h"
using namespace std;
using json = nlohmann::json;

typedef struct { vector<vector<float>> coef; vector<float> inter; } model_data;
typedef struct { vector<float> X; int y; } sample_data;

model_data load_model(string path) {
    json content = json::parse(ifstream(path));
    return {
        content["coef"].get<vector<vector<float>>>(),
        content["inter"].get<vector<float>>()
    };
}

sample_data load_sample(string path) {
    json content = json::parse(ifstream(path));
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

vector<float> dot_all(vector<vector<float>> A, vector<float> X, vector<float> b) {
    auto all = vector<float>(A.size(), 0);
    for(int i = 0; i < 10; i++)
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

model_data gradients() { /* todo */ }

int main(int argc, char** argv) {
    auto model = load_model("../sk-logistic.json");
    auto sample = load_sample(string("../dataset/") + argv[1] + ".json");
    int out = argmax(dot_all(model.coef, sample.X, model.inter));
    std::cout << out << ", should be " << sample.y << "\n";
}