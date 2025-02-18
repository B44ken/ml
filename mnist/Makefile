MNIST: json.h dataset
	g++ -O3 mnist.cpp -o MNIST

dataset:
	py download.py

json.h:
	curl -Lso json.h github.com/nlohmann/json/releases/download/v3.11.3/json.hpp

run: MNIST
	./MNIST
