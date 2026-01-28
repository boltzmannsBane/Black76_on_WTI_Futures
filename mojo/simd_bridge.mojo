# simd_bridge.mojo
# Monte Carlo bridge using Mojo's native SIMD types
# Processes 4 paths in parallel using vectorized operations

from math import sqrt, cos, log, exp
from pathlib import Path


# --- SIMD-accelerated RNG using Mojo's SIMD types ---

struct SIMD4RNG:
    """4-wide LCG random number generator using SIMD."""
    var states: SIMD[DType.uint32, 4]

    fn __init__(out self, seed: UInt32):
        # Initialize 4 independent streams with offset seeds
        self.states = SIMD[DType.uint32, 4](seed, seed + 1, seed + 2, seed + 3)

    fn next_uniform4(mut self) -> SIMD[DType.float32, 4]:
        """Generate 4 uniform random floats in [0, 1)."""
        alias a = SIMD[DType.uint32, 4](1103515245)
        alias c = SIMD[DType.uint32, 4](12345)
        alias mask = SIMD[DType.uint32, 4](0x7FFFFFFF)
        alias norm = SIMD[DType.float32, 4](1.0 / 2147483647.0)

        # LCG: state = (state * a + c) & mask
        self.states = (self.states * a + c) & mask

        # Convert to float and normalize
        return self.states.cast[DType.float32]() * norm


fn simd_box_muller(u1: SIMD[DType.float32, 4], u2: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    """Vectorized Box-Muller transform for 4 gaussians."""
    # sqrt(-2 * ln(u1)) * cos(2 * pi * u2)
    var r = SIMD[DType.float32, 4]()
    var c = SIMD[DType.float32, 4]()

    for i in range(4):
        var safe_u1 = u1[i] if u1[i] > 1e-10 else Float32(1e-10)
        r[i] = sqrt(Float32(-2.0) * log(safe_u1))
        c[i] = cos(Float32(6.283185307) * u2[i])

    return r * c


fn simd_exp4(x: SIMD[DType.float32, 4]) -> SIMD[DType.float32, 4]:
    """Vectorized exp() using Padé approximation.

    exp(x) ≈ (1 + x/2 + x²/12) / (1 - x/2 + x²/12)
    Accurate for |x| < 1.
    """
    alias half = SIMD[DType.float32, 4](0.5)
    alias one_12 = SIMD[DType.float32, 4](0.0833333333)
    alias one = SIMD[DType.float32, 4](1.0)

    var x_half = x * half        # x/2
    var x_sq = x * x             # x²
    var x_sq_12 = x_sq * one_12  # x²/12

    var num = one + x_half + x_sq_12   # 1 + x/2 + x²/12
    var den = one - x_half + x_sq_12   # 1 - x/2 + x²/12

    return num / den


struct MCResult:
    var var_99: Float64
    var cvar_99: Float64
    var mean: Float64
    var std: Float64

    fn __init__(out self, var_99: Float64, cvar_99: Float64, mean: Float64, std: Float64):
        self.var_99 = var_99
        self.cvar_99 = cvar_99
        self.mean = mean
        self.std = std


fn run_simd_monte_carlo(
    s0: Float64,
    mu: Float64,
    sigma: Float64,
    r: Float64,
    k: Float64,
    steps: Int,
    paths: Int,
) -> MCResult:
    """Run Monte Carlo with SIMD vectorization (4 paths at a time)."""

    var rng = SIMD4RNG(42)
    var payoffs = List[Float64]()

    var num_batches = (paths + 3) // 4
    var dt = 1.0 / Float64(steps)
    var drift = Float32((mu - 0.5 * sigma * sigma) * dt)
    var vol = Float32(sigma * sqrt(dt))

    alias drift_vec = SIMD[DType.float32, 4]
    alias vol_vec = SIMD[DType.float32, 4]

    for batch in range(num_batches):
        # Initialize 4 prices
        var prices = SIMD[DType.float32, 4](Float32(s0))

        # SIMD GBM simulation for all 4 paths
        for _ in range(steps):
            # Generate 8 uniforms for Box-Muller
            var u1 = rng.next_uniform4()
            var u2 = rng.next_uniform4()

            # Vectorized Box-Muller
            var z = simd_box_muller(u1, u2)

            # Vectorized GBM step: price *= exp(drift + vol * z)
            var drift_v = SIMD[DType.float32, 4](drift)
            var vol_v = SIMD[DType.float32, 4](vol)
            var exponent = drift_v + vol_v * z

            # Vectorized exp
            var mult = simd_exp4(exponent)
            prices = prices * mult

        # Compute discounted payoffs
        var discount = exp(-r * 1.0)
        for i in range(4):
            if batch * 4 + i < paths:
                var payoff = max(Float64(prices[i]) - k, 0.0) * discount
                payoffs.append(payoff)

    # Compute statistics
    var n = len(payoffs)
    var sum_val: Float64 = 0.0
    var sum_sq: Float64 = 0.0

    for i in range(n):
        sum_val += payoffs[i]
        sum_sq += payoffs[i] * payoffs[i]

    var mean = sum_val / Float64(n)
    var variance = sum_sq / Float64(n) - mean * mean
    var std = sqrt(max(variance, 0.0))

    # Sort for VaR/CVaR
    for i in range(n):
        for j in range(i + 1, n):
            if payoffs[j] < payoffs[i]:
                var tmp = payoffs[i]
                payoffs[i] = payoffs[j]
                payoffs[j] = tmp

    var var_idx = Int(Float64(n) * 0.01)
    if var_idx < 1:
        var_idx = 1
    var var_99 = payoffs[var_idx - 1]

    var cvar_sum: Float64 = 0.0
    for i in range(var_idx):
        cvar_sum += payoffs[i]
    var cvar_99 = cvar_sum / Float64(var_idx)

    return MCResult(var_99, cvar_99, mean, std)


# --- JSON I/O ---

fn is_whitespace(c: String) -> Bool:
    return c == " " or c == "\t" or c == "\n" or c == "\r"

fn is_delimiter(c: String) -> Bool:
    return c == "," or c == "}"

fn find_field_value(content: String, field: String) -> String:
    var search = '"' + field + '":'
    var idx = content.find(search)
    if idx == -1:
        return ""
    var start = idx + len(search)
    while start < len(content) and is_whitespace(String(content[start])):
        start += 1
    var end = start
    while end < len(content) and not is_delimiter(String(content[end])):
        end += 1
    return content[start:end]

fn strip_spaces(s: String) -> String:
    var result = String("")
    for i in range(len(s)):
        var c = String(s[i])
        if c != " ":
            result += c
    return result

fn parse_float(s: String) raises -> Float64:
    var cleaned = strip_spaces(s)
    var e_idx = -1
    for i in range(len(cleaned)):
        var c = String(cleaned[i])
        if c == "e" or c == "E":
            e_idx = i
            break
    if e_idx == -1:
        return atof(cleaned)
    else:
        var mantissa = atof(cleaned[0:e_idx])
        var exp_str = cleaned[e_idx + 1 : len(cleaned)]
        var exponent = atof(exp_str)
        return mantissa * (10.0 ** exponent)

fn parse_int(s: String) raises -> Int:
    return atol(strip_spaces(s))


fn main() raises:
    print("SIMD-Vectorized Monte Carlo (Mojo Native)")
    print("==========================================\n")

    # Read input.json
    var input_path = Path("input.json")
    var content = input_path.read_text()

    var s0 = parse_float(find_field_value(content, "s0"))
    var mu = parse_float(find_field_value(content, "mu"))
    var sigma = parse_float(find_field_value(content, "sigma"))
    var r = parse_float(find_field_value(content, "r"))
    var k = parse_float(find_field_value(content, "k"))
    var steps = parse_int(find_field_value(content, "steps"))
    var paths = parse_int(find_field_value(content, "paths"))

    print("Input: s0=" + String(s0) + ", k=" + String(k) + ", sigma=" + String(sigma))
    print("       steps=" + String(steps) + ", paths=" + String(paths))
    print("")

    # Run SIMD Monte Carlo
    var result = run_simd_monte_carlo(s0, mu, sigma, r, k, steps, paths)

    print("Results (SIMD 4-wide vectorized):")
    print("  Mean:   " + String(result.mean))
    print("  Std:    " + String(result.std))
    print("  VaR99:  " + String(result.var_99))
    print("  CVaR99: " + String(result.cvar_99))

    # Write output.json
    var output = '{"var_99": ' + String(result.var_99) + ', "cvar_99": ' + String(result.cvar_99) + ', "mean": ' + String(result.mean) + ', "std": ' + String(result.std) + "}"
    var output_path = Path("output.json")
    output_path.write_text(output)
    print("\nWrote output.json")
