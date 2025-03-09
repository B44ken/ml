#include "onef.h"

#define as_int(x) (x[0] > 0.5)
#define linear(m, i) (onef::Linear*)(m.pipeline[i])

void xor_eval_pretrained() {
	auto xor_net = onef::Model({
		new onef::Linear({{1, -1}, {-1, 1}}, {0, 0}),
		new onef::ReLU(),
		new onef::Linear({{1, 1}}, {0})
	});

	auto tests = onef::vec2d({{0, 0}, {0, 1}, {1, 0}, {1, 1}});
    auto results = onef::vec2d({{0}, {1}, {1}, {0}});
    auto trainer = onef::Trainer(tests, results);

    auto correct = onef::vec(4).map([&](float _, int i) {
        return (as_int(xor_net.forward(tests[i])) == as_int(results[i]));
    }).sum();

    std::cout << "ok: " << correct << "/4\n";
}

float two_input_test(int n) {
    auto model = onef::Model({
        new onef::Linear(2, n),
        new onef::ReLU(),
        new onef::Linear(n, 1)
    });
    
    auto set = onef::Trainer(
        {{-2.5, 2.0}, {9.0, 4.2}, {4.6, -9.6}, {2.0, 9.4}, {-6.9, 6.6}, {-6.9, -5.8}, {-8.8, -6.4}, {7.3, -6.3}},
        {{-5.7}, {2.9}, {-97.2}, {-97.2}, {-67.3}, {-51.3}, {-67.5}, {-27.6}}
    );

    set.train_until_convergence(&model);
    // auto first = linear(model, 0);
    return set.mse(model);
}

float linear_test() {
    auto net = onef::Model({
        new onef::Linear({{.5}}, {.5}),
        new onef::ReLU()
    });
    
    auto train = onef::Trainer(
        { {0.3}, {1.6}, {1.9}, {2.5}, {2.7} },
        { {1.1}, {1.4}, {2.0}, {2.1}, {3.0} }
    );

    train.train_epochs(&net, 100000);
    return train.mse(net);
}

int main() {
    float test = two_input_test(4);
    std::cout << "mse = " << test << "\n";
}