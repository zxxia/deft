CXX = g++

OBJ_DIR := build/obj
OBJ_HTTP_SERVER_DIR := build/obj/http_server
LIB_DIR := build/lib
BIN_DIR := build/bin
_builddirs := $(shell mkdir -p $(OBJ_DIR) $(OBJ_HTTP_SERVER_DIR) $(LIB_DIR) $(BIN_DIR))

CUDA_BIN_DIR := $(shell dirname `which nvcc`)
CUDA_INCLUDE_DIR := $(shell readlink -f $(CUDA_BIN_DIR)/../include)

# BOOST_INCLUDE_DIR = ../minor/boost_1_61_0
BOOST_INCLUDE_DIR = /usr/include/boost

# compile flags for experimenting:
CXXFLAGS = -fPIC -O3
LDFLAGS = -lcuda -lcudart -lnvidia-ml -lrt #-lpthread

OBJ_FILES := $(addprefix $(OBJ_DIR)/, hooks.o)
OBJ_FILES_SYNC := $(addprefix $(OBJ_DIR)/, hooks_sync.o)
OBJ_FILES_EVENT := $(addprefix $(OBJ_DIR)/, hooks_event.o)

HTTP_SERVER_SOURCES := $(filter-out http_server/main.cpp, $(wildcard http_server/*.cpp))
HTTP_SERVER_OBJECTS := $(patsubst %.cpp, %.o, $(HTTP_SERVER_SOURCES))
HTTP_SERVER_OBJECTS := $(addprefix $(OBJ_DIR)/, $(HTTP_SERVER_OBJECTS))

all : lib
.PHONY: all clean

lib: $(OBJ_FILES) $(OBJ_DIR)/cuda_interpose.o $(HTTP_SERVER_OBJECTS)
	$(CXX) $^ -shared -fPIC  -L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu $(LDFLAGS) -o $(LIB_DIR)/libcuinterpose.so

$(OBJ_DIR)/hooks.o: hooks.cc
	$(CXX) -I$(CUDA_INCLUDE_DIR) $(CXXFLAGS) -c -o $@ $<

$(OBJ_DIR)/cuda_interpose.o: cuda_interpose.cc
	$(CXX) -I$(CUDA_INCLUDE_DIR) $(CXXFLAGS) -c -o $@ $^

$(OBJ_HTTP_SERVER_DIR)/%.o: http_server/%.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR) $(LIB_DIR)
