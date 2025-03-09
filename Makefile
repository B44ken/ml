flag = -g -O3 -Ionef -Wall

build:
	mkdir -p build/test
	mkdir -p build/onef

clean: build
	rm -r build

build/onef/%.o: onef/%.cpp | build
	g++ $(flag) -c $< -o $@

build/test/%.o: test/%.cpp | build
	g++ $(flag) -c $< -o $@

objs := $(patsubst onef/%.cpp,build/onef/%.o,$(wildcard onef/*.cpp))

build/test/%: $(objs) build/test/%.o
	g++ $(flag) $(objs) $@.o -o build/test/$*