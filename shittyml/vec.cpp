#include <vector>
#include <string>
#include <iostream>
#include <functional>
#include "shittyml.h"


namespace shittyml {
    using namespace std;
    
    vec::vec() : vector<float>() {}
    vec::vec(initializer_list<float> list) : vector<float>(list) {}
    vec::vec(int size) : vector<float>(size, 0.f) {}

    string vec::stringify() const {
        string out = "vec(" + to_string(this->size()) + "): ";
        for (auto& val : *this)
            out += to_string(val) + ", ";
        return out.substr(0, out.size() - 2);
    }
    
    // map V[i] = f(V[i], i) over i in V
    // todo: this shouldn't modify the original
    vec vec::map(function<float(float, int)> f) {
        for (size_t i = 0; i < this->size(); i++)
            this->at(i) = f(this->at(i), i);
        return *this;
    }
    
    // map V[i] = f(V[i]) over i in V
    // todo: this shouldn't modify the original
    vec vec::map(function<float(float)> f) {
        for (size_t i = 0; i < this->size(); i++)
            this->at(i) = f(this->at(i));
        return *this;
    } 

    float vec::sum() {
        float s = 0;
        for (auto val : *this) s += val;
        return s;
    }

    ostream &operator<<(ostream &os, vec v) {
        return os << v.stringify();
    }

    vec operator+(vec v, float b) {
        return v.map([b](auto x) { return x + b; });
    }

    vec operator*(vec v, float c) {
        return v.map([c](auto x) { return x * c; });
    }
}
