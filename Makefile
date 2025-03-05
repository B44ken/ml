flag = -O3 -Ishittyml -Wall

build:
	mkdir -p build/test
	mkdir -p build/shittyml

clean: build
	rm -r build

build/shittyml/%.o: shittyml/%.cpp | build
	g++ $(flag) -c $< -o $@

build/test/%.o: test/%.cpp | build
	g++ $(flag) -c $< -o $@

objs := $(patsubst shittyml/%.cpp,build/shittyml/%.o,$(wildcard shittyml/*.cpp))

build/test/%: $(objs) build/test/%.o
	g++ $(flag) $(objs) $@.o -o build/test/$*
	./build/test/$*