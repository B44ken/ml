#include "shittyml.h"
using namespace shittyml;

vec ReLU::forward(vec a) {
    for (size_t i = 0; i < a.size(); i++)
        a[i] = a[i] > 0 ? a[i] : 0;
    return a;
}

// TODO: implement ReLU::grad
// ReLU ReLU::grad(vec a) {
//     for (size_t i = 0; i < a.size(); i++)
//         a[i] = a[i] > 0 ? 1 : 0;
//     return a;
// }