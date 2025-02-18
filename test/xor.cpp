#include "shittyml.h"

#define as_int(x) (x[0] > 0.5)

int main() {
	auto xor_net = shittyml::Model({
		new shittyml::Linear({{1, -1}, {-1, 1}}, {0, 0}),
		new shittyml::ReLU(),
		new shittyml::Linear({{1, 1}}, {0, 0})
	});

	auto tests = shittyml::vec2d({ {0, 0}, {0, 1}, {1, 0}, {1, 1} });

	for (auto test : tests) {
		int pred = as_int(xor_net.forward(test)), a = (int)test[0], b = (int)test[1];
		printf("xor(%d, %d) = %d\n", a, b, pred);
	}
}