# Black-76 Delta Hedging Simulator

A quantitative finance library implementing Black-76 option pricing, Greeks computation, delta hedging simulation, and Monte Carlo tail risk analysis. Built with **Haskell** for core pricing logic and **Mojo** for high-performance Monte Carlo simulation.

## Features

- **Black-76 Pricing**: European options on futures/forwards
- **Greeks**: Delta, Gamma, Vega (analytic closed-form)
- **Delta Hedging**: Self-financing portfolio simulation with real market data
- **Transaction Costs**: Configurable costs and rebalance thresholds
- **Realized Volatility**: Historical vol estimation with rolling windows
- **Monte Carlo**: Three simulation models:
  - GBM (Geometric Brownian Motion)
  - Jump-Diffusion (Merton model)
  - Stochastic Volatility (Heston-like)
- **Tail Risk**: VaR and CVaR (99%) computation
- **SIMD Optimization**: ARM64 NEON assembly kernels for vectorized math

## Quick Start

```bash
# Build everything
cabal build

# Run hedge simulation with real WTI futures data
cabal run wti-hedge

# Run tests (43 total: 14 property-based + 29 unit tests)
cabal test

# Run Mojo Monte Carlo
mojo run mojo/bridge.mojo

# Generate report plots
python3 plots/generate_plots.py
```

## Sample Output

```
==============================================
  WTI Futures Delta Hedge Simulation
==============================================

Loaded 1313 price points
Date range: 01/12/2021 to 01/14/2026
Price range: $52.08 - $119.78
Strike: $60.00
Final spot: $59.29
Option payoff: $0.00

--- Volatility Analysis ---
Realized volatility (historical): 35.24%
Assumed volatility: 25.00%
Vol misspecification: 10.24%

--- Hedge Simulation: No Transaction Costs ---
Final P&L: $-27.28
Rebalances: 1312 (daily)

--- Hedge Simulation: With Transaction Costs (10 bps) ---
Final P&L: $-28.71
Total transaction costs: $1.43
Rebalances: 1259

--- Monte Carlo Tail Risk Analysis ---

--- Model: GBM (baseline) ---
  Mean payoff: $5.66
  Std dev:     $9.62
  VaR (99%):   $0.00
  CVaR (99%):  $0.00

--- Model: Jump-Diffusion (Merton) ---
  Mean payoff: $7.41
  Std dev:     $13.11

--- Model: Stochastic Volatility (Heston-like) ---
  Mean payoff: $5.39
  Std dev:     $7.64
```

## Project Structure

```
BlackScholesASM/
├── src/
│   ├── Black76.hs       # Core pricing: call, put, delta, gamma, vega
│   ├── HedgeSim.hs      # Hedge simulation with tx costs & realized vol
│   ├── DataLoader.hs    # WTI futures CSV parsing
│   ├── MojoBridge.hs    # Haskell-Mojo IPC via JSON
│   └── Main.hs          # Main simulation runner
├── mojo/
│   ├── bridge.mojo      # Monte Carlo: GBM, Jump, StochVol models
│   └── simd_bridge.mojo # SIMD-vectorized Monte Carlo
├── asm/
│   ├── simd_core.s      # ARM64 NEON assembly kernels
│   ├── test_simd.c      # C test harness
│   └── Makefile         # Build for ARM64
├── test/
│   ├── Black76Tests.hs  # 14 property-based tests (Hedgehog)
│   ├── HedgeSimSpec.hs  # Transaction costs, realized vol tests
│   └── MojoBridgeSpec.hs # Monte Carlo model tests
├── plots/
│   ├── generate_plots.py # Plot generation script
│   └── *.png            # Generated visualizations
├── wti_futures.csv      # Real WTI crude oil price data
├── REPORT.md            # Comprehensive educational report
└── CLAUDE.md            # Development guide
```

## Core Modules

### Black76.hs

```haskell
-- Option pricing
black76Call  :: F -> K -> T -> r -> σ -> Price
black76Put   :: F -> K -> T -> r -> σ -> Price

-- Greeks
black76DeltaF :: F -> K -> T -> r -> σ -> Delta
black76GammaF :: F -> K -> T -> r -> σ -> Gamma
black76Vega   :: F -> K -> T -> r -> σ -> Vega
```

### HedgeSim.hs

```haskell
-- Basic hedge simulation
stepHedge :: DeltaFn -> K -> r -> σ -> [Price] -> [Time] -> [HedgeState]

-- With transaction costs and rebalance threshold
stepHedgeWithCosts :: HedgeConfig -> DeltaFn -> σ -> [Price] -> [Time] -> [HedgeState]

-- Realized volatility
realizedVol :: [Price] -> Double
rollingRealizedVol :: Window -> [Price] -> [(Int, Double)]
```

### MojoBridge.hs

```haskell
-- Model types
data ModelType = GBM | JumpDiffusion | StochasticVol

-- Run Monte Carlo with model selection
runMojoMCExt :: MojoInputExt -> IO MojoOutput
```

## Monte Carlo Models

### GBM (Geometric Brownian Motion)
Standard Black-Scholes dynamics:
```
dS = μS dt + σS dW
```

### Jump-Diffusion (Merton)
Adds Poisson jumps to GBM:
```
dS/S = (μ - λκ) dt + σ dW + (J-1) dN
```
Parameters: `jump_intensity`, `jump_mean`, `jump_vol`

### Stochastic Volatility (Heston-like)
Volatility follows its own mean-reverting process:
```
dS = μS dt + √v S dW₁
dv = κ(θ - v) dt + ξ√v dW₂
```
Parameters: `vol_mean_rev`, `vol_long_term`, `vol_of_vol`, `vol_correlation`

## Testing

```bash
# Run all tests
cabal test

# Property-based tests (Black-76 invariants)
cabal test black76-tests

# Unit tests (hedge sim, Monte Carlo models)
cabal test wti-tests
```

### Test Coverage

| Category | Tests | Description |
|----------|-------|-------------|
| Pricing | 4 | Non-negativity of call, put, gamma, vega |
| Monotonicity | 2 | Call monotone in F and σ |
| No-Arbitrage | 2 | Put-call parity, bounds |
| Greeks | 3 | Delta, gamma, vega vs numeric |
| Limits | 3 | σ→0, σ→∞, t→0 behavior |
| Transaction Costs | 4 | Cost accumulation, P&L impact |
| Rebalance Threshold | 3 | No-trade bands |
| Realized Vol | 7 | Vol estimation accuracy |
| Monte Carlo | 10 | GBM, Jump, StochVol models |

## Visualizations

The `plots/` directory contains generated visualizations:

| Plot | Description |
|------|-------------|
| `wti_prices.png` | WTI futures price history |
| `pnl_timeseries.png` | Hedge P&L comparison (4 variants) |
| `option_prices.png` | Call/put prices vs forward |
| `greeks.png` | Delta, Gamma, Vega surfaces |
| `time_decay.png` | Theta decay visualization |
| `volatility_impact.png` | Vol impact on pricing |
| `mc_convergence.png` | Monte Carlo convergence |
| `realized_vol.png` | Rolling realized vol vs assumed |
| `tx_cost_analysis.png` | Rebalancing frequency trade-off |

## Documentation

See **[REPORT.md](REPORT.md)** for a comprehensive educational guide covering:

1. No-Arbitrage Principle
2. Black-76 Model
3. Greeks (Analytic vs Numerical)
4. Finite Difference Methods
5. Delta Hedging & Self-Financing Portfolios
6. P&L Time Series
7. Rebalancing Strategies
8. Risk-Neutral Measure & Monte Carlo
9. Implied Volatility
10. Tail Risk (VaR/CVaR)
11. Practical Hedging: Transaction Costs, Vol Misspecification, Jump Risk

## Requirements

- **GHC** 9.x with Cabal
- **Mojo** 0.25.7+
- **Python 3** with matplotlib (for plots)
- **ARM64** Mac/Linux (for SIMD assembly)

## References

- Black, F. (1976). "The Pricing of Commodity Contracts"
- Hull, J. "Options, Futures, and Other Derivatives"
- Glasserman, P. "Monte Carlo Methods in Financial Engineering"

## License

MIT
