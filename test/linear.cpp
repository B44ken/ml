#include "shittyml.h"

int main() {
    auto model = shittyml::Model({
        new shittyml::Linear(1, 1)
    });

    auto mult_two = shittyml::Trainer(
        { {0.3}, {1.6}, {1.9}, {2.5} },
        { 1.1,    1.4,   2.0,   2.1  }
    );

    std::cout << "loss before\t" << mult_two.mean_sq_error(model) << "\n";
    mult_two.train_epochs(&model, 10000);
    std::cout << "loss after\t" << mult_two.mean_sq_error(model) << "\n";
}