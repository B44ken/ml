.SILENT: test

<<<<<<< Updated upstream
XOR: json.h xor.cpp boilerplate.cpp
	echo g++ xor.cpp
	g++ -O3 xor.cpp -o XOR

json.h:
	echo curl json.h
=======
XOR: json.h xor.cpp boilerplate.h
	g++ -O3 xor.cpp -o XOR

json.h:
>>>>>>> Stashed changes
	curl -Lso json.h github.com/nlohmann/json/releases/download/v3.11.3/json.hpp

run: XOR
	./XOR

test: XOR
	echo `./XOR 0 0` `./XOR 0 1` `./XOR 1 0` `./XOR 1 1` > /tmp/xortest
	echo '0 1 1 0' | diff - /tmp/xortest
	echo Pass
