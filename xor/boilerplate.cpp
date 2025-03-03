#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include "json.h"
using namespace std;
using json = nlohmann::json;

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

float dlt(int a, int b) { return (a == b) ? 1 : 0; }
