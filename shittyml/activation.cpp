#include "shittyml.h"
using namespace shittyml;

vec ReLU::forward(vec a) {
    for (size_t i = 0; i < a.size(); i++)
        a[i] = a[i] > 0 ? a[i] : 0;
    return a;
}