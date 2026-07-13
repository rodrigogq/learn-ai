# Optional C++ comparison for Chapter 8 (the course itself stays in C).
# Build:  make -C chapters/08-backpropagation/cpp
# Run:    ./chapters/08-backpropagation/cpp/build/tiny_autograd

CXX ?= c++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -O2

build/tiny_autograd: tiny_autograd.cpp | build
	$(CXX) $(CXXFLAGS) tiny_autograd.cpp -o build/tiny_autograd

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: clean
