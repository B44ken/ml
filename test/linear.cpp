#include "shittyml.h"

void xor_eval_pretrained() {
    #define as_int(x) (x[0] > 0.5)

	auto xor_net = shittyml::Model({
		new shittyml::Linear({{1, -1}, {-1, 1}}, {0, 0}),
		new shittyml::ReLU(),
		new shittyml::Linear({{1, 1}}, {0})
	});

	auto tests = shittyml::vec2d({ {0, 0}, {0, 1}, {1, 0}, {1, 1} });

    for (size_t i = 0; i < tests.size(); i++) {
		int pred = as_int(xor_net.forward(tests[i]));
        int a = (int)tests[i][0], b = (int)tests[i][1];
		printf("xor(%d, %d) = %d (exp %d)\n", a, b, pred, a^b);
	}
}

void linear_simple_train() {
    auto model = shittyml::Model({
        new shittyml::Linear(1, 1),
        new shittyml::Sigmoid(),
    });

    auto set = shittyml::Trainer(
        { {0.3}, {1.6}, {1.9}, {2.5} },
        {  1.1,   1.4,   2.0,   2.1  }
    );

    std::cout << "loss = " << set.mse(model) << "\n";
    set.train_until_convergence(&model);
    std::cout << "loss = " << set.mse(model) << "\n";
}

void xor_train_2331() {
    auto model = shittyml::Model({
        new shittyml::Linear(2, 3),
        new shittyml::Sigmoid(),
        new shittyml::Linear(3, 3),
        new shittyml::Sigmoid(),
        new shittyml::Linear(2, 1)
    });

    auto y = shittyml::vec({ 0, 1, 1, 0 });
    auto X = shittyml::vec2d({ {0, 0}, {0, 1}, {1, 0}, {1, 1} });
    auto set = shittyml::Trainer(X, y);

    std::cout << "loss = " << set.mse(model) << "\n";
    set.train_epochs(&model, 1000*1000);
    std::cout << "loss = " << set.mse(model) << "\n";
}

int main() {
    linear_simple_train();
}