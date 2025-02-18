#include <vector>
#include <string>
#include <iostream>
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

    std::ostream &operator<<(std::ostream &os, shittyml::vec v) {
        return os << v.stringify();
    }
}
