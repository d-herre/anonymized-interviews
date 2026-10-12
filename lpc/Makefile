CXX ?= g++
CC ?= gcc
CXXFLAGS ?= -std=c++17 -O2 -Wall
CFLAGS ?= -std=c99 -O2 -Wall
SOURCES = rt_lpc.cpp lpc.cpp psola.cpp audio_io.cpp silk_analysis.cpp
HEADERS = lpc.h psola.h audio_io.h silk_analysis.h silk_port.h

rt_lpc: $(SOURCES) $(HEADERS) silk_port.o
	$(CXX) $(CXXFLAGS) -o $@ $(SOURCES) silk_port.o -pthread

silk_port.o: silk_port.c silk_port.h
	$(CC) $(CFLAGS) -c -o $@ silk_port.c

clean:
	rm -f rt_lpc rt_lpc_debug silk_port.o silk_port_debug.o

.PHONY: clean

# rt_lpc with AddressSanitizer and UBSan: slower, but if it crashes its
# stderr (which the media server logs) says where and why
rt_lpc_debug: $(SOURCES) $(HEADERS) silk_port.c
	$(CC) -std=c99 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -c -o silk_port_debug.o silk_port.c
	$(CXX) -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -o $@ $(SOURCES) silk_port_debug.o -pthread
