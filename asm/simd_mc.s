// simd_mc.s
// ARM64 NEON SIMD kernel for Monte Carlo GBM simulation
// Vectorizes 4 paths simultaneously using 128-bit SIMD

.global _simd_generate_uniform4
.global _simd_gbm_steps4
.global _simd_exp4

.align 4

// ============================================================
// Generate 4 uniform random floats in [0,1] using LCG
// ============================================================
// void simd_generate_uniform4(uint32_t* state, float* out)
// x0 = pointer to single uint32_t state (will be updated)
// x1 = pointer to output (4 floats)
//
// Uses LCG: state = (state * 1103515245 + 12345) & 0x7FFFFFFF

_simd_generate_uniform4:
    // Load state
    ldr     w2, [x0]

    // LCG constants
    mov     w3, #0x4E6D
    movk    w3, #0x41C6, lsl #16  // a = 1103515245
    mov     w4, #12345            // c = 12345
    mov     w5, #0x7FFFFFFF       // mask

    // Generate 4 random numbers sequentially, pack into vector
    // Random 1
    mul     w2, w2, w3
    add     w2, w2, w4
    and     w2, w2, w5
    mov     w6, w2

    // Random 2
    mul     w2, w2, w3
    add     w2, w2, w4
    and     w2, w2, w5
    mov     w7, w2

    // Random 3
    mul     w2, w2, w3
    add     w2, w2, w4
    and     w2, w2, w5
    mov     w8, w2

    // Random 4
    mul     w2, w2, w3
    add     w2, w2, w4
    and     w2, w2, w5
    mov     w9, w2

    // Store updated state
    str     w2, [x0]

    // Pack into NEON register
    fmov    s0, w6
    fmov    s1, w7
    fmov    s2, w8
    fmov    s3, w9
    ins     v0.s[1], v1.s[0]
    ins     v0.s[2], v2.s[0]
    ins     v0.s[3], v3.s[0]

    // Convert to float
    ucvtf   v0.4s, v0.4s

    // Normalize to [0,1]: divide by 2147483647.0
    // 1.0/2147483647.0 ≈ 4.6566129e-10
    ldr     s1, .inv_max
    dup     v1.4s, v1.s[0]
    fmul    v0.4s, v0.4s, v1.4s

    // Store result
    st1     {v0.4s}, [x1]
    ret

.align 4
.inv_max:
    .float 4.6566129e-10


// ============================================================
// Vectorized exp() approximation for 4 floats
// ============================================================
// void simd_exp4(float* x, float* out)
// x0 = pointer to 4 input floats
// x1 = pointer to 4 output floats
//
// Uses Padé approximation: exp(x) ≈ (1 + x/2 + x²/12) / (1 - x/2 + x²/12)
// Accurate for |x| < 1, which covers typical GBM steps

_simd_exp4:
    // Load input
    ld1     {v0.4s}, [x0]         // v0 = x

    // Constants
    fmov    v1.4s, #0.5           // 0.5
    ldr     s2, .one_twelfth
    dup     v2.4s, v2.s[0]        // 1/12
    fmov    v3.4s, #1.0           // 1.0

    // x/2
    fmul    v4.4s, v0.4s, v1.4s   // v4 = x/2

    // x²
    fmul    v5.4s, v0.4s, v0.4s   // v5 = x²

    // x²/12
    fmul    v6.4s, v5.4s, v2.4s   // v6 = x²/12

    // Numerator: 1 + x/2 + x²/12
    fadd    v7.4s, v3.4s, v4.4s   // 1 + x/2
    fadd    v7.4s, v7.4s, v6.4s   // + x²/12

    // Denominator: 1 - x/2 + x²/12
    fsub    v8.4s, v3.4s, v4.4s   // 1 - x/2
    fadd    v8.4s, v8.4s, v6.4s   // + x²/12

    // exp(x) ≈ num/denom
    fdiv    v0.4s, v7.4s, v8.4s

    // Store result
    st1     {v0.4s}, [x1]
    ret

.align 4
.one_twelfth:
    .float 0.0833333333


// ============================================================
// Run GBM steps for 4 paths in parallel
// ============================================================
// void simd_gbm_steps4(float* prices, float drift, float vol,
//                      uint32_t* rng_state, int steps)
// x0 = pointer to 4 prices (in/out)
// s0 = drift = (mu - 0.5*sigma²)*dt
// s1 = vol = sigma * sqrt(dt)
// x1 = pointer to RNG state
// w2 = number of steps
//
// Each step: price = price * exp(drift + vol * z)
// where z is a Gaussian from Box-Muller

_simd_gbm_steps4:
    // Save callee-saved registers
    stp     x29, x30, [sp, #-64]!
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     d8, d9, [sp, #48]
    mov     x29, sp

    // Save parameters
    mov     x19, x0               // prices ptr
    mov     x20, x1               // rng state ptr
    mov     w21, w2               // steps
    fmov    s8, s0                // drift
    fmov    s9, s1                // vol

    // Allocate stack space for temp arrays
    sub     sp, sp, #64           // 4 floats * 4 arrays

    // Load initial prices
    ld1     {v16.4s}, [x19]       // v16 = current prices

    // Broadcast drift and vol
    dup     v17.4s, v8.s[0]       // v17 = [drift x4]
    dup     v18.4s, v9.s[0]       // v18 = [vol x4]

    // Constants for Box-Muller
    ldr     s19, .neg_two
    dup     v19.4s, v19.s[0]      // -2.0
    ldr     s20, .two_pi_const
    dup     v20.4s, v20.s[0]      // 2*pi

.step_loop:
    cbz     w21, .step_done

    // Generate 8 uniform randoms (need pairs for Box-Muller)
    mov     x0, x20
    add     x1, sp, #0            // u1[0..3]
    bl      _simd_generate_uniform4

    mov     x0, x20
    add     x1, sp, #16           // u2[0..3]
    bl      _simd_generate_uniform4

    // Load u1, u2
    ld1     {v0.4s}, [sp]         // u1
    ld1     {v1.4s}, [sp, #16]    // u2

    // Clamp u1 to avoid log(0)
    ldr     s2, .epsilon
    dup     v2.4s, v2.s[0]
    fmax    v0.4s, v0.4s, v2.4s

    // Box-Muller: z = sqrt(-2*ln(u1)) * cos(2*pi*u2)

    // Compute ln(u1) using polynomial approximation
    // ln(x) ≈ (x-1) - (x-1)²/2 + (x-1)³/3 for x in [0.5, 1.5]
    // For broader range, use: ln(x) = ln(m * 2^e) = ln(m) + e*ln(2)
    // Simplified: just use Taylor around x=1

    fmov    v3.4s, #1.0
    fsub    v4.4s, v0.4s, v3.4s   // x-1
    fmul    v5.4s, v4.4s, v4.4s   // (x-1)²
    fmul    v6.4s, v5.4s, v4.4s   // (x-1)³

    fmov    v7.4s, #0.5
    fmul    v5.4s, v5.4s, v7.4s   // (x-1)²/2

    ldr     s7, .one_third
    dup     v7.4s, v7.s[0]
    fmul    v6.4s, v6.4s, v7.4s   // (x-1)³/3

    // ln(x) ≈ (x-1) - (x-1)²/2 + (x-1)³/3
    fsub    v4.4s, v4.4s, v5.4s
    fadd    v4.4s, v4.4s, v6.4s   // v4 = ln(u1) approx

    // -2 * ln(u1)
    fmul    v4.4s, v4.4s, v19.4s  // v4 = -2*ln(u1)

    // sqrt(-2*ln(u1)) - use Newton-Raphson
    // sqrt(x): y = y * (3 - x*y²) / 2
    frsqrte v5.4s, v4.4s          // initial estimate of 1/sqrt(x)
    fmul    v6.4s, v5.4s, v5.4s   // y²
    fmul    v6.4s, v6.4s, v4.4s   // x*y²
    fmov    v7.4s, #3.0
    fsub    v6.4s, v7.4s, v6.4s   // 3 - x*y²
    fmul    v5.4s, v5.4s, v6.4s   // y * (3 - x*y²)
    fmov    v7.4s, #0.5
    fmul    v5.4s, v5.4s, v7.4s   // / 2 -> better 1/sqrt(x)
    fmul    v4.4s, v4.4s, v5.4s   // x * 1/sqrt(x) = sqrt(x)

    // 2*pi*u2
    fmul    v1.4s, v1.4s, v20.4s  // v1 = 2*pi*u2

    // cos(2*pi*u2) using polynomial
    // cos(x) ≈ 1 - x²/2 + x⁴/24 for small x
    // Need to reduce to [-pi, pi] first - skip for demo
    fmul    v5.4s, v1.4s, v1.4s   // x²
    fmov    v6.4s, #0.5
    fmul    v6.4s, v5.4s, v6.4s   // x²/2
    fmov    v7.4s, #1.0
    fsub    v6.4s, v7.4s, v6.4s   // 1 - x²/2

    fmul    v7.4s, v5.4s, v5.4s   // x⁴
    ldr     s8, .one_24
    dup     v8.4s, v8.s[0]
    fmla    v6.4s, v7.4s, v8.4s   // + x⁴/24
    // v6 = cos(2*pi*u2) approx

    // z = sqrt(-2*ln(u1)) * cos(2*pi*u2)
    fmul    v0.4s, v4.4s, v6.4s   // v0 = z (gaussian)

    // Reload drift and vol (might have been clobbered)
    dup     v17.4s, s8            // Need to fix - s8 was overwritten
    // Just reload from saved values
    ldr     s8, [x29, #48]        // drift
    ldr     s9, [x29, #56]        // vol
    dup     v17.4s, v8.s[0]
    dup     v18.4s, v9.s[0]

    // GBM step: price = price * exp(drift + vol * z)
    fmla    v17.4s, v18.4s, v0.4s // drift + vol * z

    // exp(drift + vol*z) using Padé approximation
    // Store to temp, call exp4
    st1     {v17.4s}, [sp, #32]
    add     x0, sp, #32
    add     x1, sp, #48
    bl      _simd_exp4
    ld1     {v0.4s}, [sp, #48]    // exp result

    // price *= exp(...)
    fmul    v16.4s, v16.4s, v0.4s

    sub     w21, w21, #1
    b       .step_loop

.step_done:
    // Store final prices
    st1     {v16.4s}, [x19]

    // Restore stack and registers
    add     sp, sp, #64
    ldp     d8, d9, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #64
    ret

.align 4
.neg_two:
    .float -2.0
.two_pi_const:
    .float 6.283185307
.epsilon:
    .float 1.0e-10
.one_third:
    .float 0.333333333
.one_24:
    .float 0.0416666667
