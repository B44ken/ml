INCL = -I./shittyml

build:
	mkdir -p build

test_xor: build
	g++ -O3 -Wall $(INCL) test/xor.cpp shittyml/*.cpp -o build/test

test_xor_run: test_xor
	./build/test