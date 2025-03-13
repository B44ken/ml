#include "onef.h"

#define as_int(x) (x[0] > 0.5)
#define linear(m, i) (onef::Linear*)(m.pipeline[i])

float xor_eval_pretrained_test();
float two_input_test(int n);
float linear_behind_noop_test();
float linear_test();

float onef::lr = 0.0001;
int main(int argc, char** argv) {

    std::string test = argc > 1 ? argv[1] : "linear_test";

    float loss = 0;
    if(test == "xor")
        loss = xor_eval_pretrained_test();
    else if(test == "2-input")
        loss = two_input_test(3);
    else if(test == "linear")
        loss = linear_test();
    else if(test == "noop-linear")
        loss = linear_behind_noop_test();
    else
        std::cout << "unknown test" << test << std::endl;

    std::cout << loss << std::endl;
}

float xor_eval_pretrained_test() {
	auto xor_net = onef::Model({
		new onef::Linear({{1, -1}, {-1, 1}}, {0, 0}),
		new onef::ReLU(),
		new onef::Linear({{1, 1}}, {0})
	});

	auto tests = onef::vec2d({{0, 0}, {0, 1}, {1, 0}, {1, 1}});
    auto results = onef::vec2d({{0}, {1}, {1}, {0}});

    auto correct = onef::vec(4).map([&](float _, int i) {
        return as_int(xor_net.forward(tests[i])) == as_int(results[i]);
    }).sum();

    return correct/4;
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
    return set.mse(model);
}

float linear_test() {
    auto net = onef::Model({
        new onef::Linear(1, 1),
    });
    
    auto train = onef::Trainer(
        { {0.3}, {1.6}, {1.9}, {2.5}, {2.7} },
        { {1.1}, {1.4}, {2.0}, {2.1}, {3.0} }
    );

    train.train_epochs(&net, 100000);
    return train.mse(net);
}

float linear_behind_noop_test() {
    auto net = onef::Model({
        new onef::Linear(1, 1),
        new onef::NoOpLayer(),
        new onef::NoOpLayer(),
        new onef::NoOpLayer()
    });
    
    auto train = onef::Trainer(
        { {0.3}, {1.6}, {1.9}, {2.5}, {2.7} },
        { {1.1}, {1.4}, {2.0}, {2.1}, {3.0} }
    );

    train.train_epochs(&net, 100000);
    return train.mse(net);
}
