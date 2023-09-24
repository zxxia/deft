CXX = g++

OBJ_DIR := build/obj
OBJ_HTTP_SERVER_DIR := build/obj/http_server
LIB_DIR := build/lib
BIN_DIR := build/bin
_builddirs := $(shell mkdir -p $(OBJ_DIR) $(OBJ_HTTP_SERVER_DIR) $(LIB_DIR) $(BIN_DIR))

CUDA_BIN_DIR := $(shell dirname `which nvcc`)
CUDA_INCLUDE_DIR := $(shell readlink -f $(CUDA_BIN_DIR)/../include)

# BOOST_INCLUDE_DIR = ../minor/boost_1_61_0
BOOST_INCLUDE_DIR := /usr/include/boost

CXXFLAGS := -fPIC -O3
LDFLAGS := -lcuda -lcudart -lnvidia-ml -lrt #-lpthread
DEPFLAGS := -MMD -MP

WARNING := -Wall -Wextra

SOURCES := $(wildcard *.cpp) $(filter-out http_server/main.cpp, $(wildcard http_server/*.cpp))
OBJECTS := $(addprefix $(OBJ_DIR)/, $(patsubst %.cpp, %.o, $(SOURCES)))
DEPENDS := $(addprefix $(OBJ_DIR)/, $(patsubst %.cpp, %.d, $(SOURCES)))

all: lib
.PHONY: all clean lib

clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR) $(LIB_DIR)

lib: $(OBJECTS)
	$(CXX) $^ -shared -fPIC -L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu $(LDFLAGS) -o $(LIB_DIR)/libcuinterpose.so

$(OBJ_DIR)/%.o: %.cpp Makefile
	$(CXX) $(WARNING) -I$(CUDA_INCLUDE_DIR) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

$(OBJ_HTTP_SERVER_DIR)/%.o: http_server/%.cpp Makefile
	$(CXX) $(WARNING) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

-include $(DEPENDS)
