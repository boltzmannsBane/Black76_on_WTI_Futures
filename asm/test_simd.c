// test_simd.c
// Test harness for NEON SIMD kernels

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

// Assembly function declarations
extern void neon_lcg4(uint32_t* state, float* out);
extern void neon_exp4(float* x, float* out);
extern void neon_mul4(float* a, float* b, float* out);
extern void neon_fma4(float* a, float* b, float* c, float* out);

// Scalar reference implementations
float scalar_lcg(uint32_t* state) {
    *state = (*state * 1103515245 + 12345) & 0x7FFFFFFF;
    return (float)(*state) / 2147483647.0f;
}

float scalar_exp_pade(float x) {
    float x2 = x * x;
    float num = 1.0f + x/2.0f + x2/12.0f;
    float den = 1.0f - x/2.0f + x2/12.0f;
    return num / den;
}

void test_lcg4() {
    printf("=== Testing neon_lcg4 ===\n");

    uint32_t state_simd = 42;
    uint32_t state_scalar = 42;
    float out_simd[4];

    neon_lcg4(&state_simd, out_simd);

    printf("SIMD output: [%.6f, %.6f, %.6f, %.6f]\n",
           out_simd[0], out_simd[1], out_simd[2], out_simd[3]);

    printf("Scalar check: ");
    for (int i = 0; i < 4; i++) {
        float s = scalar_lcg(&state_scalar);
        printf("%.6f ", s);
    }
    printf("\n");

    printf("States match: %s\n\n",
           state_simd == state_scalar ? "YES" : "NO");
}

void test_exp4() {
    printf("=== Testing neon_exp4 ===\n");

    float x[4] = {0.0f, 0.1f, -0.1f, 0.5f};
    float out_simd[4];

    neon_exp4(x, out_simd);

    printf("Input:       [%.4f, %.4f, %.4f, %.4f]\n",
           x[0], x[1], x[2], x[3]);
    printf("SIMD exp:    [%.6f, %.6f, %.6f, %.6f]\n",
           out_simd[0], out_simd[1], out_simd[2], out_simd[3]);
    printf("stdlib exp:  [%.6f, %.6f, %.6f, %.6f]\n",
           expf(x[0]), expf(x[1]), expf(x[2]), expf(x[3]));
    printf("Padé ref:    [%.6f, %.6f, %.6f, %.6f]\n\n",
           scalar_exp_pade(x[0]), scalar_exp_pade(x[1]),
           scalar_exp_pade(x[2]), scalar_exp_pade(x[3]));
}

void test_mul4() {
    printf("=== Testing neon_mul4 ===\n");

    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4] = {2.0f, 3.0f, 4.0f, 5.0f};
    float out[4];

    neon_mul4(a, b, out);

    printf("a * b = [%.1f, %.1f, %.1f, %.1f]\n\n",
           out[0], out[1], out[2], out[3]);
}

void test_fma4() {
    printf("=== Testing neon_fma4 ===\n");

    float a[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float b[4] = {2.0f, 2.0f, 2.0f, 2.0f};
    float c[4] = {3.0f, 4.0f, 5.0f, 6.0f};
    float out[4];

    neon_fma4(a, b, c, out);

    printf("a + b*c = [%.1f, %.1f, %.1f, %.1f]\n",
           out[0], out[1], out[2], out[3]);
    printf("Expected:   [7.0, 9.0, 11.0, 13.0]\n\n");
}

void benchmark_lcg() {
    printf("=== Benchmark: LCG ===\n");

    const int ITERATIONS = 10000000;
    float out[4];
    volatile float sink = 0;  // prevent optimization
    uint32_t state;
    clock_t start, end;

    // Scalar benchmark
    state = 42;
    start = clock();
    for (int i = 0; i < ITERATIONS; i++) {
        sink += scalar_lcg(&state);
        sink += scalar_lcg(&state);
        sink += scalar_lcg(&state);
        sink += scalar_lcg(&state);
    }
    end = clock();
    double scalar_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Scalar (4x%d): %.3f sec (sink=%.2f)\n", ITERATIONS, scalar_time, sink);

    // SIMD benchmark
    state = 42;
    sink = 0;
    start = clock();
    for (int i = 0; i < ITERATIONS; i++) {
        neon_lcg4(&state, out);
        sink += out[0] + out[1] + out[2] + out[3];
    }
    end = clock();
    double simd_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("SIMD   (4x%d): %.3f sec (sink=%.2f)\n", ITERATIONS, simd_time, sink);

    printf("Speedup: %.2fx\n\n", scalar_time / simd_time);
}

void benchmark_exp() {
    printf("=== Benchmark: exp ===\n");

    const int ITERATIONS = 10000000;
    float x[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    float out[4];
    volatile float sink = 0;
    clock_t start, end;

    // Scalar benchmark (stdlib)
    start = clock();
    for (int i = 0; i < ITERATIONS; i++) {
        sink += expf(x[0]);
        sink += expf(x[1]);
        sink += expf(x[2]);
        sink += expf(x[3]);
    }
    end = clock();
    double scalar_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Scalar expf (4x%d): %.3f sec\n", ITERATIONS, scalar_time);

    // SIMD benchmark
    sink = 0;
    start = clock();
    for (int i = 0; i < ITERATIONS; i++) {
        neon_exp4(x, out);
        sink += out[0] + out[1] + out[2] + out[3];
    }
    end = clock();
    double simd_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("SIMD exp   (4x%d): %.3f sec\n", ITERATIONS, simd_time);

    printf("Speedup: %.2fx\n\n", scalar_time / simd_time);
}

int main() {
    printf("NEON SIMD Monte Carlo Kernel Tests\n");
    printf("===================================\n\n");

    test_lcg4();
    test_exp4();
    test_mul4();
    test_fma4();

    printf("--- Benchmarks ---\n\n");
    benchmark_lcg();
    benchmark_exp();

    return 0;
}
