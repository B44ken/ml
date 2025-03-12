# onef
im learning ml and built a machine learning framework in c++. 

im calling it `1f4a9`, or `onef` for short.

if you want to use it you can go to [onef (b44ken/ml)](https://github.com/B44ken/ml/tree/main/onef). you can use it like

```c++
#include "onef.h"

auto model = onef::Model({
    new onef::Linear(1, 1)
});

auto set = onef::Trainer(
    {{0.3}, {1.6}, {1.9}, {2.5}},
    {{1.1}, {1.4}, {2.0}, {2.1}}
);

std::cout << "loss =\t" << set.mse(model) << "\n";
set.train_epochs(&model, 10000);
std::cout << "loss =\t" << set.mse(model) << "\n";
```

to do list

- [x] backprop
- [ ] import weights from json
- [ ] cuda acceleration
- [ ] world domination
