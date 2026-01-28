// simd_core.s
// ARM64 NEON SIMD core operations for Monte Carlo
// Clean, minimal kernels that can be composed

.global _neon_lcg4
.global _neon_exp4
.global _neon_mul4
.global _neon_fma4

.text
.align 4

// ============================================================
// 4-wide LCG: generate 4 uniform floats from scalar state
// ============================================================
// void neon_lcg4(uint32_t* state, float* out)
// Generates 4 sequential LCG values, returns as floats in [0,1)

_neon_lcg4:
    ldr     w2, [x0]              // load state

    // LCG constants: a=1103515245, c=12345
    mov     w3, #0x4E6D
    movk    w3, #0x41C6, lsl #16
    mov     w4, #12345
    mov     w5, #0x7FFFFFFF

    // Generate 4 values
    mul     w2, w2, w3
    add     w2, w2, w4
    and     w6, w2, w5            // r0

    mul     w2, w2, w3
    add     w2, w2, w4
    and     w7, w2, w5            // r1

    mul     w2, w2, w3
    add     w2, w2, w4
    and     w8, w2, w5            // r2

    mul     w2, w2, w3
    add     w2, w2, w4
    and     w9, w2, w5            // r3

    str     w2, [x0]              // save state

    // Pack into vector register
    mov     v0.s[0], w6
    mov     v0.s[1], w7
    mov     v0.s[2], w8
    mov     v0.s[3], w9

    // Convert to float [0,1)
    ucvtf   v0.4s, v0.4s
    ldr     s1, .Linv_max
    dup     v1.4s, v1.s[0]
    fmul    v0.4s, v0.4s, v1.4s

    st1     {v0.4s}, [x1]
    ret

.align 4
.Linv_max:
    .float 4.6566129e-10


// ============================================================
// 4-wide exp() using Padé approximation
// ============================================================
// void neon_exp4(float* x, float* out)
// exp(x) ≈ (1 + x/2 + x²/12) / (1 - x/2 + x²/12)
// Good for |x| < 1

_neon_exp4:
    ld1     {v0.4s}, [x0]         // x

    fmov    v1.4s, #0.5           // 0.5
    ldr     s2, .Lone_12
    dup     v2.4s, v2.s[0]        // 1/12
    fmov    v3.4s, #1.0           // 1.0

    fmul    v4.4s, v0.4s, v1.4s   // x/2
    fmul    v5.4s, v0.4s, v0.4s   // x²
    fmul    v6.4s, v5.4s, v2.4s   // x²/12

    // num = 1 + x/2 + x²/12
    fadd    v7.4s, v3.4s, v4.4s
    fadd    v7.4s, v7.4s, v6.4s

    // den = 1 - x/2 + x²/12
    fsub    v8.4s, v3.4s, v4.4s
    fadd    v8.4s, v8.4s, v6.4s

    fdiv    v0.4s, v7.4s, v8.4s

    st1     {v0.4s}, [x1]
    ret

.align 4
.Lone_12:
    .float 0.0833333333


// ============================================================
// 4-wide multiply
// ============================================================
// void neon_mul4(float* a, float* b, float* out)

_neon_mul4:
    ld1     {v0.4s}, [x0]
    ld1     {v1.4s}, [x1]
    fmul    v0.4s, v0.4s, v1.4s
    st1     {v0.4s}, [x2]
    ret


// ============================================================
// 4-wide fused multiply-add: out = a + b * c
// ============================================================
// void neon_fma4(float* a, float* b, float* c, float* out)

_neon_fma4:
    ld1     {v0.4s}, [x0]         // a
    ld1     {v1.4s}, [x1]         // b
    ld1     {v2.4s}, [x2]         // c
    fmla    v0.4s, v1.4s, v2.4s   // a + b*c
    st1     {v0.4s}, [x3]
    ret
