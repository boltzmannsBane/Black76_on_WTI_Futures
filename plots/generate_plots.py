#!/usr/bin/env python3
"""
Generate plots for the Black-76 Delta Hedging report.
"""

import csv
import math
import os
from pathlib import Path

# Try to import matplotlib, install if needed
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("Installing matplotlib...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'matplotlib'])
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

# Ensure plots directory exists
os.makedirs('plots', exist_ok=True)

# ============================================================
# Black-76 Implementation (Python version for plotting)
# ============================================================

def norm_cdf(x):
    """Standard normal CDF using error function approximation."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

def norm_pdf(x):
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def black76_call(f, k, t, r, sigma):
    """Black-76 call price."""
    if t <= 0:
        return max(0, f - k)
    denom = sigma * math.sqrt(t)
    d1 = (math.log(f / k) + 0.5 * sigma * sigma * t) / denom
    d2 = d1 - denom
    return max(0, math.exp(-r * t) * (f * norm_cdf(d1) - k * norm_cdf(d2)))

def black76_delta(f, k, t, r, sigma):
    """Black-76 delta."""
    if t <= 0:
        return math.exp(-r * t) * (1 if f > k else 0)
    denom = sigma * math.sqrt(t)
    d1 = (math.log(f / k) + 0.5 * sigma * sigma * t) / denom
    return math.exp(-r * t) * norm_cdf(d1)

def black76_gamma(f, k, t, r, sigma):
    """Black-76 gamma."""
    if t <= 0:
        return 0
    denom = sigma * math.sqrt(t)
    d1 = (math.log(f / k) + 0.5 * sigma * sigma * t) / denom
    return math.exp(-r * t) * norm_pdf(d1) / (f * sigma * math.sqrt(t))

def black76_vega(f, k, t, r, sigma):
    """Black-76 vega."""
    if t <= 0:
        return 0
    denom = sigma * math.sqrt(t)
    d1 = (math.log(f / k) + 0.5 * sigma * sigma * t) / denom
    return math.exp(-r * t) * f * math.sqrt(t) * norm_pdf(d1)

# ============================================================
# Plot 1: P&L Time Series from Hedge Simulation
# ============================================================

def plot_pnl_timeseries():
    """Plot P&L over time from the hedge simulation."""
    pnl_file = 'pnl.csv'
    if not os.path.exists(pnl_file):
        print(f"Warning: {pnl_file} not found. Run 'cabal run wti-hedge' first.")
        return

    steps = []
    pnls = []
    with open(pnl_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['step']))
            pnls.append(float(row['pnl']))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, pnls, 'b-', linewidth=0.8, alpha=0.8)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.fill_between(steps, pnls, 0, where=[p > 0 for p in pnls],
                    color='green', alpha=0.3, label='Profit')
    ax.fill_between(steps, pnls, 0, where=[p < 0 for p in pnls],
                    color='red', alpha=0.3, label='Loss')

    ax.set_xlabel('Trading Day', fontsize=11)
    ax.set_ylabel('Hedge P&L ($)', fontsize=11)
    ax.set_title('Delta Hedge P&L Time Series (WTI Futures, Daily Rebalancing)', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Add summary stats
    final_pnl = pnls[-1] if pnls else 0
    max_pnl = max(pnls) if pnls else 0
    min_pnl = min(pnls) if pnls else 0
    stats_text = f'Final P&L: ${final_pnl:.2f}\nMax: ${max_pnl:.2f}\nMin: ${min_pnl:.2f}'
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('plots/pnl_timeseries.png', dpi=150)
    plt.close()
    print("Created plots/pnl_timeseries.png")

# ============================================================
# Plot 2: WTI Price History
# ============================================================

def plot_wti_prices():
    """Plot WTI futures price history."""
    wti_file = 'wti_futures.csv'
    if not os.path.exists(wti_file):
        print(f"Warning: {wti_file} not found.")
        return

    dates = []
    prices = []
    with open(wti_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dates.append(row['Date'].strip('"'))
            prices.append(float(row['Price'].strip('"')))

    # Reverse to chronological order
    dates = dates[::-1]
    prices = prices[::-1]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(prices)), prices, 'b-', linewidth=0.8)

    # Mark key levels
    ax.axhline(y=60, color='r', linestyle='--', linewidth=1, alpha=0.7, label='Strike K=$60')

    ax.set_xlabel('Trading Day', fontsize=11)
    ax.set_ylabel('WTI Futures Price ($)', fontsize=11)
    ax.set_title(f'WTI Crude Oil Futures ({dates[0]} to {dates[-1]})', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add price range annotation
    stats_text = f'High: ${max(prices):.2f}\nLow: ${min(prices):.2f}\nFinal: ${prices[-1]:.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('plots/wti_prices.png', dpi=150)
    plt.close()
    print("Created plots/wti_prices.png")

# ============================================================
# Plot 3: Option Price vs Underlying (varying F)
# ============================================================

def plot_option_price_vs_underlying():
    """Plot call and put prices as functions of underlying price."""
    k = 60  # Strike
    t = 1.0  # 1 year
    r = 0.05
    sigma = 0.25

    forwards = [f for f in range(30, 100)]
    calls = [black76_call(f, k, t, r, sigma) for f in forwards]
    puts = [black76_call(f, k, t, r, sigma) - math.exp(-r*t)*(f - k) for f in forwards]  # From put-call parity
    intrinsic = [max(0, f - k) for f in forwards]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(forwards, calls, 'b-', linewidth=2, label='Call Price')
    ax.plot(forwards, puts, 'r-', linewidth=2, label='Put Price')
    ax.plot(forwards, intrinsic, 'g--', linewidth=1.5, alpha=0.7, label='Intrinsic Value')
    ax.axvline(x=k, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.text(k+1, max(calls)*0.9, f'K={k}', fontsize=10, color='gray')

    ax.set_xlabel('Forward Price ($)', fontsize=11)
    ax.set_ylabel('Option Price ($)', fontsize=11)
    ax.set_title(f'Black-76 Option Prices (K=${k}, T=1yr, σ={sigma*100:.0f}%, r={r*100:.0f}%)', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(30, 100)
    ax.set_ylim(0, None)

    plt.tight_layout()
    plt.savefig('plots/option_prices.png', dpi=150)
    plt.close()
    print("Created plots/option_prices.png")

# ============================================================
# Plot 4: Greeks vs Underlying
# ============================================================

def plot_greeks():
    """Plot Delta, Gamma, Vega as functions of underlying price."""
    k = 60
    t = 1.0
    r = 0.05
    sigma = 0.25

    forwards = [f for f in range(30, 100)]
    deltas = [black76_delta(f, k, t, r, sigma) for f in forwards]
    gammas = [black76_gamma(f, k, t, r, sigma) for f in forwards]
    vegas = [black76_vega(f, k, t, r, sigma) for f in forwards]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Delta
    axes[0].plot(forwards, deltas, 'b-', linewidth=2)
    axes[0].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    axes[0].axvline(x=k, color='gray', linestyle=':', alpha=0.5)
    axes[0].set_ylabel('Delta (Δ)', fontsize=11)
    axes[0].set_title('Greeks vs Forward Price', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    axes[0].text(k+1, 0.5, 'ATM', fontsize=9, color='gray')

    # Gamma
    axes[1].plot(forwards, gammas, 'g-', linewidth=2)
    axes[1].axvline(x=k, color='gray', linestyle=':', alpha=0.5)
    axes[1].set_ylabel('Gamma (Γ)', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Vega
    axes[2].plot(forwards, vegas, 'r-', linewidth=2)
    axes[2].axvline(x=k, color='gray', linestyle=':', alpha=0.5)
    axes[2].set_ylabel('Vega (ν)', fontsize=11)
    axes[2].set_xlabel('Forward Price ($)', fontsize=11)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/greeks.png', dpi=150)
    plt.close()
    print("Created plots/greeks.png")

# ============================================================
# Plot 5: Option Price vs Time to Expiry
# ============================================================

def plot_time_decay():
    """Plot option price as time approaches expiry."""
    k = 60
    r = 0.05
    sigma = 0.25

    times = [t/100 for t in range(1, 101)]  # 0.01 to 1.0 years

    # Different moneyness levels
    f_itm = 70  # In the money
    f_atm = 60  # At the money
    f_otm = 50  # Out of the money

    calls_itm = [black76_call(f_itm, k, t, r, sigma) for t in times]
    calls_atm = [black76_call(f_atm, k, t, r, sigma) for t in times]
    calls_otm = [black76_call(f_otm, k, t, r, sigma) for t in times]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, calls_itm, 'g-', linewidth=2, label=f'ITM (F=${f_itm})')
    ax.plot(times, calls_atm, 'b-', linewidth=2, label=f'ATM (F=${f_atm})')
    ax.plot(times, calls_otm, 'r-', linewidth=2, label=f'OTM (F=${f_otm})')

    # Show intrinsic values at expiry
    ax.axhline(y=f_itm-k, color='g', linestyle=':', alpha=0.5)
    ax.axhline(y=0, color='r', linestyle=':', alpha=0.5)

    ax.set_xlabel('Time to Expiry (years)', fontsize=11)
    ax.set_ylabel('Call Price ($)', fontsize=11)
    ax.set_title(f'Time Decay (Theta): Call Price vs Time (K=${k}, σ={sigma*100:.0f}%)', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig('plots/time_decay.png', dpi=150)
    plt.close()
    print("Created plots/time_decay.png")

# ============================================================
# Plot 6: Volatility Impact
# ============================================================

def plot_volatility_impact():
    """Plot option price as function of volatility."""
    f = 60  # ATM
    k = 60
    t = 1.0
    r = 0.05

    sigmas = [s/100 for s in range(5, 81)]  # 5% to 80%
    calls = [black76_call(f, k, t, r, s) for s in sigmas]
    vegas = [black76_vega(f, k, t, r, s) for s in sigmas]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Price vs Vol
    axes[0].plot([s*100 for s in sigmas], calls, 'b-', linewidth=2)
    axes[0].set_ylabel('Call Price ($)', fontsize=11)
    axes[0].set_title(f'Volatility Impact on ATM Option (F=K=${k}, T=1yr)', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Vega vs Vol
    axes[1].plot([s*100 for s in sigmas], vegas, 'r-', linewidth=2)
    axes[1].set_ylabel('Vega (ν)', fontsize=11)
    axes[1].set_xlabel('Implied Volatility (%)', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/volatility_impact.png', dpi=150)
    plt.close()
    print("Created plots/volatility_impact.png")

# ============================================================
# Plot 7: Monte Carlo Convergence
# ============================================================

def plot_mc_convergence():
    """Illustrate Monte Carlo convergence (simulated)."""
    import random
    random.seed(42)

    # Simulate MC convergence for option pricing
    f, k, t, r, sigma = 60, 60, 1, 0.05, 0.25
    true_price = black76_call(f, k, t, r, sigma)

    path_counts = [10, 50, 100, 500, 1000, 5000, 10000, 50000]
    estimates = []
    std_errors = []

    for n in path_counts:
        payoffs = []
        dt = t
        drift = -0.5 * sigma * sigma * t
        vol = sigma * math.sqrt(t)

        for _ in range(n):
            z = random.gauss(0, 1)
            st = f * math.exp(drift + vol * z)
            payoff = max(st - k, 0) * math.exp(-r * t)
            payoffs.append(payoff)

        mean = sum(payoffs) / n
        var = sum((p - mean)**2 for p in payoffs) / n
        se = math.sqrt(var / n)

        estimates.append(mean)
        std_errors.append(se)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(path_counts, estimates, yerr=[1.96*se for se in std_errors],
                fmt='o-', capsize=5, capthick=1.5, linewidth=2, markersize=8,
                label='MC Estimate ± 95% CI')
    ax.axhline(y=true_price, color='r', linestyle='--', linewidth=2,
               label=f'Black-76 Price (${true_price:.2f})')

    ax.set_xscale('log')
    ax.set_xlabel('Number of Paths', fontsize=11)
    ax.set_ylabel('Option Price ($)', fontsize=11)
    ax.set_title('Monte Carlo Convergence', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/mc_convergence.png', dpi=150)
    plt.close()
    print("Created plots/mc_convergence.png")

# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("Generating plots for Black-76 Delta Hedging Report...")
    print("=" * 50)

    plot_wti_prices()
    plot_pnl_timeseries()
    plot_option_price_vs_underlying()
    plot_greeks()
    plot_time_decay()
    plot_volatility_impact()
    plot_mc_convergence()

    print("=" * 50)
    print("Done! Plots saved to plots/ directory.")
