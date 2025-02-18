build: json.h
	g++ -O3 mnist.cpp -o MNIST

json.h:
	curl -Lso json.h github.com/nlohmann/json/releases/download/v3.11.3/json.hpp

run: build
	./MNIST
