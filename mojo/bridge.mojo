# mojo/bridge.mojo
# Mojo bridge for Haskell <-> Monte Carlo simulation
# Reads input.json, runs MC, writes output.json with VaR/CVaR
#
# Supports:
#   - Standard GBM (Geometric Brownian Motion)
#   - Jump-diffusion (Merton model)
#   - Stochastic volatility (simplified Heston-like)

from math import sqrt, exp, log, cos
from pathlib import Path


# --- Random Number Generator ---

struct RNG:
    var state: Int

    fn __init__(out self, seed: Int):
        self.state = seed

    fn next_float(mut self) -> Float64:
        # Linear Congruential Generator
        self.state = (self.state * 1103515245 + 12345) & 0x7FFFFFFF
        return Float64(self.state) / 2147483647.0

    fn next_gaussian(mut self) -> Float64:
        # Box-Muller transform
        var u1 = self.next_float()
        var u2 = self.next_float()
        if u1 < 1e-10:
            u1 = 1e-10
        return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.141592653589793 * u2)

    fn next_poisson(mut self, lambda_: Float64) -> Int:
        """Generate Poisson random variable using inverse transform."""
        var L = exp(-lambda_)
        var k = 0
        var p = 1.0
        while p > L:
            k += 1
            p *= self.next_float()
        return k - 1


# --- Result struct ---

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


# --- Monte Carlo Engines ---

fn simulate_path_gbm(
    s0: Float64,
    mu: Float64,
    sigma: Float64,
    r: Float64,
    k: Float64,
    steps: Int,
    mut rng: RNG,
) -> Float64:
    """Simulate one GBM path and return discounted call payoff."""
    var dt = 1.0 / Float64(steps)
    var price = s0

    # GBM: dS = mu*S*dt + sigma*S*dW
    var drift = (mu - 0.5 * sigma * sigma) * dt
    var vol = sigma * sqrt(dt)

    for _ in range(steps):
        var z = rng.next_gaussian()
        price = price * exp(drift + vol * z)

    # European call payoff, discounted
    var payoff = max(price - k, 0.0)
    return exp(-r * 1.0) * payoff


fn simulate_path_jump_diffusion(
    s0: Float64,
    mu: Float64,
    sigma: Float64,
    r: Float64,
    k: Float64,
    steps: Int,
    jump_intensity: Float64,  # lambda: expected jumps per year
    jump_mean: Float64,       # mean of log jump size
    jump_vol: Float64,        # volatility of log jump size
    mut rng: RNG,
) -> Float64:
    """
    Simulate one jump-diffusion (Merton) path.

    Model: dS/S = (mu - lambda*kappa) dt + sigma dW + (J-1) dN

    where:
      - N is Poisson process with intensity lambda
      - J is lognormal: log(J) ~ N(jump_mean, jump_vol^2)
      - kappa = E[J-1] = exp(jump_mean + 0.5*jump_vol^2) - 1
    """
    var dt = 1.0 / Float64(steps)
    var price = s0

    # Compensator for jump (risk-neutral drift adjustment)
    var kappa = exp(jump_mean + 0.5 * jump_vol * jump_vol) - 1.0

    # Adjusted drift for jump compensation
    var drift = (mu - jump_intensity * kappa - 0.5 * sigma * sigma) * dt
    var vol = sigma * sqrt(dt)

    # Expected number of jumps per step
    var lambda_dt = jump_intensity * dt

    for _ in range(steps):
        # Diffusion component
        var z = rng.next_gaussian()
        price = price * exp(drift + vol * z)

        # Jump component: Poisson number of jumps
        var num_jumps = rng.next_poisson(lambda_dt)

        # Apply each jump
        for _ in range(num_jumps):
            var jump_z = rng.next_gaussian()
            var log_jump = jump_mean + jump_vol * jump_z
            price = price * exp(log_jump)

    # European call payoff, discounted
    var payoff = max(price - k, 0.0)
    return exp(-r * 1.0) * payoff


fn simulate_path_stochastic_vol(
    s0: Float64,
    mu: Float64,
    sigma: Float64,      # Initial volatility (v0)
    r: Float64,
    k: Float64,
    steps: Int,
    vol_mean_rev: Float64,  # kappa: mean reversion speed
    vol_long_term: Float64, # theta: long-term volatility
    vol_of_vol: Float64,    # xi: volatility of volatility
    vol_correlation: Float64,  # rho: correlation between price and vol
    mut rng: RNG,
) -> Float64:
    """
    Simulate one stochastic volatility path (simplified Heston-like).

    Price: dS = mu*S dt + sqrt(v)*S dW1
    Vol:   dv = kappa*(theta - v) dt + xi*sqrt(v) dW2

    with correlation: dW1 * dW2 = rho * dt
    """
    var dt = 1.0 / Float64(steps)
    var price = s0
    var vol = sigma * sigma  # Start with variance = sigma^2

    for _ in range(steps):
        # Generate correlated Gaussians
        var z1 = rng.next_gaussian()
        var z2_ind = rng.next_gaussian()
        var z2 = vol_correlation * z1 + sqrt(1.0 - vol_correlation * vol_correlation) * z2_ind

        # Ensure variance stays positive (reflection)
        var v_safe = max(vol, 1e-8)
        var sqrt_v = sqrt(v_safe)

        # Price update
        var price_drift = (mu - 0.5 * v_safe) * dt
        var price_vol = sqrt_v * sqrt(dt)
        price = price * exp(price_drift + price_vol * z1)

        # Variance update (Euler-Maruyama)
        var dv = vol_mean_rev * (vol_long_term * vol_long_term - v_safe) * dt + vol_of_vol * sqrt_v * sqrt(dt) * z2
        vol = max(vol + dv, 1e-8)  # Floor at small positive value

    # European call payoff, discounted
    var payoff = max(price - k, 0.0)
    return exp(-r * 1.0) * payoff


fn run_monte_carlo(
    s0: Float64,
    mu: Float64,
    sigma: Float64,
    r: Float64,
    k: Float64,
    steps: Int,
    paths: Int,
    model: String,
    # Jump parameters (used if model == "jump")
    jump_intensity: Float64,
    jump_mean: Float64,
    jump_vol: Float64,
    # Stochastic vol parameters (used if model == "stochvol")
    vol_mean_rev: Float64,
    vol_long_term: Float64,
    vol_of_vol: Float64,
    vol_correlation: Float64,
) -> MCResult:
    """Run MC simulation and return VaR, CVaR, mean, std."""
    var rng = RNG(42)

    # Collect all payoffs
    var payoffs = List[Float64]()

    for _ in range(paths):
        var pnl: Float64
        if model == "jump":
            pnl = simulate_path_jump_diffusion(
                s0, mu, sigma, r, k, steps,
                jump_intensity, jump_mean, jump_vol, rng
            )
        elif model == "stochvol":
            pnl = simulate_path_stochastic_vol(
                s0, mu, sigma, r, k, steps,
                vol_mean_rev, vol_long_term, vol_of_vol, vol_correlation, rng
            )
        else:  # Default: GBM
            pnl = simulate_path_gbm(s0, mu, sigma, r, k, steps, rng)
        payoffs.append(pnl)

    # Compute mean and std
    var sum_val: Float64 = 0.0
    var sum_sq: Float64 = 0.0
    for i in range(len(payoffs)):
        sum_val += payoffs[i]
        sum_sq += payoffs[i] * payoffs[i]

    var mean = sum_val / Float64(paths)
    var variance = sum_sq / Float64(paths) - mean * mean
    var std = sqrt(max(variance, 0.0))

    # Sort payoffs for VaR/CVaR (ascending order)
    for i in range(len(payoffs)):
        for j in range(i + 1, len(payoffs)):
            if payoffs[j] < payoffs[i]:
                var tmp = payoffs[i]
                payoffs[i] = payoffs[j]
                payoffs[j] = tmp

    # VaR at 99% = 1st percentile of P&L (worst 1%)
    var var_idx = Int(Float64(paths) * 0.01)
    if var_idx < 1:
        var_idx = 1
    var var_99 = payoffs[var_idx - 1]

    # CVaR = expected value of worst 1%
    var cvar_sum: Float64 = 0.0
    for i in range(var_idx):
        cvar_sum += payoffs[i]
    var cvar_99 = cvar_sum / Float64(var_idx)

    return MCResult(var_99, cvar_99, mean, std)


# Legacy function for backward compatibility
fn run_monte_carlo_simple(
    s0: Float64,
    mu: Float64,
    sigma: Float64,
    r: Float64,
    k: Float64,
    steps: Int,
    paths: Int,
) -> MCResult:
    """Run MC simulation with GBM (backward compatible)."""
    return run_monte_carlo(
        s0, mu, sigma, r, k, steps, paths,
        "gbm",  # model
        0.0, 0.0, 0.0,  # jump params (unused)
        0.0, 0.0, 0.0, 0.0  # stochvol params (unused)
    )


# --- JSON Parsing (minimal manual parser) ---

fn is_whitespace(c: String) -> Bool:
    return c == " " or c == "\t" or c == "\n" or c == "\r"


fn is_delimiter(c: String) -> Bool:
    return c == "," or c == "}"


fn find_field_value(content: String, field: String) -> String:
    """Extract a field value from JSON string."""
    var search = '"' + field + '":'
    var idx = content.find(search)
    if idx == -1:
        return ""

    var start = idx + len(search)

    # Skip whitespace
    while start < len(content) and is_whitespace(String(content[start])):
        start += 1

    # Read until comma or closing brace
    var end = start
    while end < len(content) and not is_delimiter(String(content[end])):
        end += 1

    return content[start:end]


fn find_field_string(content: String, field: String) -> String:
    """Extract a string field value from JSON (with quotes)."""
    var raw = find_field_value(content, field)
    # Remove surrounding quotes
    if len(raw) >= 2 and String(raw[0]) == '"':
        return raw[1:len(raw)-1]
    return raw


fn strip_spaces(s: String) -> String:
    """Remove leading/trailing spaces."""
    var result = String("")
    for i in range(len(s)):
        var c = String(s[i])
        if c != " ":
            result += c
    return result


fn parse_float(s: String) raises -> Float64:
    """Parse a float from string, handling scientific notation."""
    var cleaned = strip_spaces(s)
    if len(cleaned) == 0:
        return 0.0

    # Handle scientific notation like 1.0e-2
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
    """Parse an integer from string."""
    var cleaned = strip_spaces(s)
    if len(cleaned) == 0:
        return 0
    return atol(cleaned)


fn main() raises:
    # Read input.json
    var input_path = Path("input.json")
    var content = input_path.read_text()

    # Parse core input parameters
    var s0 = parse_float(find_field_value(content, "s0"))
    var mu = parse_float(find_field_value(content, "mu"))
    var sigma = parse_float(find_field_value(content, "sigma"))
    var r = parse_float(find_field_value(content, "r"))
    var k = parse_float(find_field_value(content, "k"))
    var steps = parse_int(find_field_value(content, "steps"))
    var paths = parse_int(find_field_value(content, "paths"))

    # Parse model type (default: "gbm")
    var model = find_field_string(content, "model")
    if len(model) == 0:
        model = "gbm"

    # Parse jump-diffusion parameters (optional)
    var jump_intensity = parse_float(find_field_value(content, "jump_intensity"))
    var jump_mean = parse_float(find_field_value(content, "jump_mean"))
    var jump_vol = parse_float(find_field_value(content, "jump_vol"))

    # Parse stochastic volatility parameters (optional)
    var vol_mean_rev = parse_float(find_field_value(content, "vol_mean_rev"))
    var vol_long_term = parse_float(find_field_value(content, "vol_long_term"))
    var vol_of_vol = parse_float(find_field_value(content, "vol_of_vol"))
    var vol_correlation = parse_float(find_field_value(content, "vol_correlation"))

    # Run Monte Carlo
    var result = run_monte_carlo(
        s0, mu, sigma, r, k, steps, paths,
        model,
        jump_intensity, jump_mean, jump_vol,
        vol_mean_rev, vol_long_term, vol_of_vol, vol_correlation
    )

    # Write output.json
    var output = '{"var_99": ' + String(result.var_99) + ', "cvar_99": ' + String(result.cvar_99) + ', "mean": ' + String(result.mean) + ', "std": ' + String(result.std) + ', "model": "' + model + '"}'

    var output_path = Path("output.json")
    output_path.write_text(output)
