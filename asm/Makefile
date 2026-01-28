# Makefile for NEON SIMD Monte Carlo kernels
# Targets ARM64 (Apple M1/M2/M3)

CC = clang
AS = clang
CFLAGS = -O3 -Wall -arch arm64
ASFLAGS = -arch arm64

.PHONY: all clean test lib

all: test_simd libsimd_mc.dylib

# Assemble the SIMD core
simd_core.o: simd_core.s
	$(AS) $(ASFLAGS) -c $< -o $@

# Compile test harness
test_simd.o: test_simd.c
	$(CC) $(CFLAGS) -c $< -o $@

# Link test executable
test_simd: test_simd.o simd_core.o
	$(CC) $(CFLAGS) $^ -o $@

# Build shared library for Mojo FFI
libsimd_mc.dylib: simd_core.o
	$(CC) -shared -arch arm64 $< -o $@

# Run tests
test: test_simd
	./test_simd

clean:
	rm -f *.o test_simd libsimd_mc.dylib
