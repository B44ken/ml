#pragma once

#include <numeric>
#include <vector>
#include <cmath>
#include <iostream>
#include "json.h"
using namespace std;
using json = nlohmann::json;

typedef vector<float> vec;
typedef vector<vector<float>> mat;

typedef struct {
    mat W; vec b; string tag;
} layer_data;
typedef vector<layer_data> model_data;

typedef struct {
    vec X; int y;
} class_data;

class linear {
    mat W;
    vec b;
    vec forward(vec x) {
        return logit(vec(W.size(), 0), W, b);
    }
};

layer_data do_relu = {{}, {}, "do_relu"};

int argmax(vector<float> a) {
    int best = 0;
    for(int i = 0; i < a.size(); i++)
        if(a[i] > a[best]) best = i;
    return best;
}

float dot(vec a, vec b) { return inner_product(a.begin(), a.end(), b.begin(), 0.f); }

// float tanh(float x) {
//     return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
// }
//

vec scale(float x, vec v) {
    for(int i = 0; i < v.size(); i++)
        v[i] *= x;
    return v;
}

vec sum(vec a, vec b) {
    for(int i = 0; i < a.size(); i++)
        a[i] *= b[i];
    return a;
}

vec logit(vec X, mat W, vec b) {
    auto all = vec(W.size(), 0);
    for(int i = 0; i < W.size(); i++)
        all[i] = dot(W[i], X) + b[i];
    return all;
}

vec logit(vec X, layer_data l) {
    return logit(X, l.W, l.b);
}

vec tanh(vec x) {
    for(int i = 0; i < x.size(); i++)
        x[i] = tanh(x[i]);
    return x;
}

vec relu(vec x) {
    for(int i = 0; i < x.size(); i++)
        x[i] = max(0.f, x[i]);
    return x;
}

vec softmax(vec a) {
    float denom = 0;
    for(int i = 0; i < a.size(); i++)
        denom += exp(a[i]);

    auto softed = vec(a.size(), 0);
    for(int i = 0; i < a.size(); i++)
        softed[i] = exp(a[i]) / denom;

    return softed;
}

// compute xA + b
float linear(vec x, vec A, float b) {
    float sum = 0;
    for(int i = 0; i < A.size(); i++)
        sum += x[i] * A[i];
    return sum + b;
}

// compute x_iA_i + b_i for all i in list
vec linear_multi(vec x, mat W, vec b) {
    vec sums = vec(W.size(), 0.f);
    for(int i = 0; i < W.size(); i++)
        sums[i] = linear(x, W[i], b[i]);
    return sums;
}

string stringify(mat M) {
    string buf;
    buf += "mat(" + to_string(M.size()) + ")\n";
    for(int i = 0; i < M.size(); i++) {
        for(int j = 0; j < M[i].size(); j++)
            buf += to_string(M[i][j]) + " ";
        buf += "\n";
    }
    return buf;
}

string stringify(vec v) {
    string buf;
    buf += "vec(" + to_string(v.size()) + ")\n";
    for(int i = 0; i < v.size(); i++)
        buf += to_string(v[i]) + " ";
    buf += "\n";
    return buf;
}

float dlt(int a, int b) { return (a == b) ? 1 : 0; }
