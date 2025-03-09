# onef
im learning ml and built a machine learning framework in c++. 

im calling it `1f4a9`, or `onef` for short.

if you want to use it you can go to shittyml (b44ken/ml). you can use it like

#include "onef.h"

auto model = shittyml::Model({
    new shittyml::Linear(1, 1)
});

auto set = shittyml::Trainer(
    {{0.3}, {1.6}, {1.9}, {2.5}},
    {{1.1}, {1.4}, {2.0}, {2.1}}
);

std::cout << "loss =\t" << set.mse(model) << "\n";
set.train_epochs(&model, 10000);
std::cout << "loss =\t" << set.mse(model) << "\n";

to do list

- [x] backprop
- [ ] import weights from json
- [ ] cuda acceleration
- [ ] world domination