CXX := g++
SRC := server
SOURCES := $(wildcard $(SRC)/*.cpp)
OBJECTS := $(patsubst $(SRC)/%.cpp, $(SRC)/%.o, $(SOURCES))
.PHONY: all clean
# all:
# 	g++ -lboost_system server/*.cpp -lpthread -L/usr/lib/x86_64-linux-gnu -o http_server

all: http_server

http_server: $(OBJECTS)
	$(CXX) $^ -L/usr/lib/x86_64-linux-gnu -lpthread -o $@

# all: server/connection.o server/connection_manager.o server/mime_types.o \
# 	server/reply.o server/request_handler.o server/request_parser.o \
# 	server/server.o
# Define a pattern rule that compiles every .c file into a .o file
# %.o : %.cpp
# 	$(CXX) -c $(CFLAGS) $(CPPFLAGS) $< -o $@
clean:
	rm http_server server/*.o
