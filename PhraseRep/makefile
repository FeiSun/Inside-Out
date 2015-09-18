CXX = g++
CXXFLAGS = -I/usr/local/include/eigen/ -std=c++11 -O3 -march=native -funroll-loops
LDFLAGS = -lm -fopenmp

all: p2v

p2v : main.cpp PhraseRep.cpp
	$(CXX) main.cpp PhraseRep.cpp -o p2v $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -rf p2v
