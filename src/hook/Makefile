CXX = g++

OBJ_DIR := build/obj
LIB_DIR := build/lib
BIN_DIR := build/bin
_builddirs := $(shell mkdir -p $(OBJ_DIR) $(LIB_DIR) $(BIN_DIR))

CUDA_BIN_DIR := $(shell dirname `which nvcc`)
CUDA_INCLUDE_DIR := $(shell readlink -f $(CUDA_BIN_DIR)/../include)

# BOOST_INCLUDE_DIR = ../minor/boost_1_61_0
BOOST_INCLUDE_DIR = /usr/include/boost

# compile flags for experimenting:
CXXFLAGS = -fPIC -O3 -D_SCHEDULER_LOCK
LDFLAGS = -lcuda -lcudart -lnvidia-ml -lrt -lpthread

OBJ_FILES := $(addprefix $(OBJ_DIR)/, hooks.o)
OBJ_FILES_SYNC := $(addprefix $(OBJ_DIR)/, hooks_sync.o)
OBJ_FILES_EVENT := $(addprefix $(OBJ_DIR)/, hooks_event.o)
all : lib libsync libevent
.PHONY: all clean

lib: $(OBJ_FILES) $(OBJ_DIR)/cuda_interpose.o
	$(CXX) $^ -shared -fPIC  -L/usr/local/cuda/lib64 $(LDFLAGS) -o $(LIB_DIR)/libcuinterpose.so

libsync: $(OBJ_FILES_SYNC) $(OBJ_DIR)/cuda_interpose.o
	$(CXX) $^ -shared -fPIC  -L/usr/local/cuda/lib64 $(LDFLAGS) -o $(LIB_DIR)/libcuinterpose_sync.so

libevent: $(OBJ_FILES_EVENT) $(OBJ_DIR)/cuda_interpose.o
	$(CXX) $^ -shared -fPIC  -L/usr/local/cuda/lib64 $(LDFLAGS) -o $(LIB_DIR)/libcuinterpose_event.so

$(OBJ_DIR)/hooks.o: hooks.cc
	$(CXX) -I$(CUDA_INCLUDE_DIR) -I$(BOOST_INCLUDE_DIR) $(CXXFLAGS) -c -o $@ $<

$(OBJ_DIR)/hooks_sync.o: hooks.cc
	$(CXX) -I$(CUDA_INCLUDE_DIR) -I$(BOOST_INCLUDE_DIR) $(CXXFLAGS) -D_SYNC_QUEUE -c -o $@ $<

$(OBJ_DIR)/hooks_event.o: hooks.cc
	$(CXX) -I$(CUDA_INCLUDE_DIR) -I$(BOOST_INCLUDE_DIR) $(CXXFLAGS) -D_GROUP_EVENT -c -o $@ $<

$(OBJ_DIR)/cuda_interpose.o: cuda_interpose.cc
	$(CXX) -I$(CUDA_INCLUDE_DIR) $(CXXFLAGS) -c -o $@ $^

clean:
	rm -f $(OBJ_DIR)/* $(LIB_DIR)/libcuinterpose.so $(LIB_DIR)/libcuinterpose_sync.so $(LIB_DIR)/libcuinterpose_event.so
