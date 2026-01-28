# mojo/bridge.mojo
# Mojo bridge for Haskell <-> Monte Carlo simulation
# Reads input.json, runs MC, writes output.json with VaR/CVaR

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


# --- Monte Carlo Engine ---

fn simulate_path(
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


fn run_monte_carlo(
    s0: Float64,
    mu: Float64,
    sigma: Float64,
    r: Float64,
    k: Float64,
    steps: Int,
    paths: Int,
) -> MCResult:
    """Run MC simulation and return VaR, CVaR, mean, std."""
    var rng = RNG(42)

    # Collect all payoffs
    var payoffs = List[Float64]()
    for _ in range(paths):
        var pnl = simulate_path(s0, mu, sigma, r, k, steps, rng)
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
    return atol(strip_spaces(s))


fn main() raises:
    # Read input.json
    var input_path = Path("input.json")
    var content = input_path.read_text()

    # Parse input parameters
    var s0 = parse_float(find_field_value(content, "s0"))
    var mu = parse_float(find_field_value(content, "mu"))
    var sigma = parse_float(find_field_value(content, "sigma"))
    var r = parse_float(find_field_value(content, "r"))
    var k = parse_float(find_field_value(content, "k"))
    var steps = parse_int(find_field_value(content, "steps"))
    var paths = parse_int(find_field_value(content, "paths"))

    # Run Monte Carlo
    var result = run_monte_carlo(s0, mu, sigma, r, k, steps, paths)

    # Write output.json
    var output = '{"var_99": ' + String(result.var_99) + ', "cvar_99": ' + String(result.cvar_99) + ', "mean": ' + String(result.mean) + ', "std": ' + String(result.std) + "}"

    var output_path = Path("output.json")
    output_path.write_text(output)
