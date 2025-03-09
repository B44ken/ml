#include "shittyml.h"


namespace shittyml {
    using namespace std;

    vec::vec() : vector<float>() {}
    vec::vec(initializer_list<float> list) : vector<float>(list) {}
    vec::vec(int size) : vector<float>(size, 0.f) {}

    string vec::stringify() const {
        string out = "[";
        for(auto& val : *this)
            out += to_string(val).substr(0, 5) + " ";
        return out.substr(0, out.size() - 1) + "]";
        // string out = "vec(" + to_string(this->size()) + "): ";
        // for (auto& val : *this)
        //     out += to_string(val) + ", ";
        // return out.substr(0, out.size() - 2);
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

    ostream& operator<<(ostream& os, vec v) {
        return os << v.stringify();
    }

    vec operator+(vec v, float b) {
        return v.map([b](auto x) { return x + b; });
    }

    vec operator*(vec v, float c) {
        return v.map([c](auto x) { return x * c; });
    }

    float operator*(vec a, vec b) {
        return vec(a).map([b](float x, int i) { return x * b[i]; }).sum();
    }

    vec2d::vec2d(initializer_list<initializer_list<float>> rows) {
        for (auto row : rows) {
            this->push_back(vec(row));
        }
    }

    vec2d::vec2d() : vector<vec>() {}
    vec2d::vec2d(int c, int r) : vector<vec>(c, vec(r)) {}

    vec2d vec2d::transpose() {
        vec2d out(this->at(0).size(), this->size());
        for (size_t i = 0; i < this->size(); i++) {
            for (size_t j = 0; j < this->at(i).size(); j++) {
                out[j][i] = this->at(i)[j];
            }
        }
        return out;
    }

    vec operator*(vec2d A, vec x) {
        A = A.transpose();
        vec out(A.size());
        for (size_t i = 0; i < A.size(); i++) {
            float dot = A[i] * x;
            out[i] = dot;
        }
        return out;
    }

    ostream& operator<<(ostream& os, vec2d m) {
        os << "vec2d\n";
        for (auto row : m)
            os << row << "\n";
        return os;
    }
}
