CXX = g++
CXXFLAGS = -lrt -lpthread
INCLUDE_DIR = ../minor/boost_1_61_0

all: expset.o expset.so

.PHONY: all
expset.o: expset.cc
	$(CXX) -c -fPIC -I $(INCLUDE_DIR) $< $(CXXFLAGS) -o $@

expset.so: expset.o
	$(CXX) -shared -fPIC -Wl,-soname,$@ -I $(INCLUDE_DIR) $< $(CXXFLAGS) -o $@

.PHONY: clean
clean:
	rm expset.o expset.so
