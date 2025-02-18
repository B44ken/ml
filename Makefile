INCL = -I./shittyml

build:
	mkdir -p build

# shittyml.o: build shittyml/*.cpp shittyml/*.h
# 	g++ -c -O3 shittyml/*.cpp -o build/shittyml.o

test: build
	g++ -O3 $(INCL) test/test.cpp shittyml/*.cpp -o build/test

run_test: test
	./build/test

xor: build shittyml.o
	g++ -O3 $(INCL) xor/*.cpp build/shittyml.o -o build/xor