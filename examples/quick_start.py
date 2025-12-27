#!/usr/bin/env python3
"""
Quick Start Example - Black-Scholes Greek Surface Visualization

This script demonstrates the basic functionality of the package.
Run this script to generate Greek surface visualizations.
"""

import sys
sys.path.insert(0, '..')

from src.black_scholes import BlackScholes, verify_bs_pde
from src.visualizations import GreekSurfaceVisualizer
from src.static_plots import StaticGreekPlotter


def example_pricing():
    """Demonstrate option pricing and Greeks calculation."""
    print("=" * 60)
    print("Example 1: Option Pricing and Greeks")
    print("=" * 60)

    # Create a Black-Scholes instance for an ATM call option
    bs = BlackScholes(
        S=100,      # Spot price
        K=100,      # Strike price (ATM)
        T=1.0,      # 1 year to expiration
        r=0.05,     # 5% risk-free rate
        sigma=0.2   # 20% volatility
    )

    print(f"\nOption Parameters:")
    print(f"  Spot (S):     ${bs.S}")
    print(f"  Strike (K):   ${bs.K}")
    print(f"  Time (T):     {bs.T} year")
    print(f"  Rate (r):     {bs.r:.1%}")
    print(f"  Vol (σ):      {bs.sigma:.1%}")

    print(f"\nIntermediate Values:")
    print(f"  d₁ = {bs.d1:.6f}")
    print(f"  d₂ = {bs.d2:.6f}")

    print(f"\nOption Prices:")
    print(f"  Call: ${bs.call_price():.4f}")
    print(f"  Put:  ${bs.put_price():.4f}")

    print(f"\nFirst-Order Greeks (Call):")
    print(f"  Delta (Δ): {bs.delta('call'):.6f}")
    print(f"  Gamma (Γ): {bs.gamma():.6f}")
    print(f"  Theta (Θ): {bs.theta('call'):.6f} per year")
    print(f"  Vega (V):  {bs.vega():.6f}")
    print(f"  Rho (ρ):   {bs.rho('call'):.6f}")

    # Verify BS PDE
    lhs, rhs = verify_bs_pde(100, 100, 1.0, 0.05, 0.2)
    print(f"\nBS PDE Verification: Θ + ½σ²S²Γ + rSΔ = rV")
    print(f"  LHS: {lhs:.6f}")
    print(f"  RHS: {rhs:.6f}")
    print(f"  Error: {abs(lhs - rhs):.2e}")


def example_interactive_visualization():
    """Create interactive Plotly visualizations."""
    print("\n" + "=" * 60)
    print("Example 2: Interactive Visualization")
    print("=" * 60)

    viz = GreekSurfaceVisualizer(K=100, r=0.05, sigma=0.2)

    # Create Delta surface
    print("\nCreating interactive Delta surface...")
    fig = viz.plot_greek_surface(
        greek='delta',
        option_type='call',
        current_S=100,
        current_T=0.5,
        show_atm_line=True,
        show_peak=True
    )
    viz.save_html(fig, 'delta_surface.html')
    print("  Saved: delta_surface.html")

    # Create all Greeks
    print("\nCreating all Greeks visualization...")
    fig_all = viz.plot_all_greeks(
        option_type='call',
        current_S=100,
        current_T=0.5
    )
    viz.save_html(fig_all, 'all_greeks.html')
    print("  Saved: all_greeks.html")

    print("\nOpen the HTML files in your browser to explore!")


def example_static_visualization():
    """Create static matplotlib visualizations."""
    print("\n" + "=" * 60)
    print("Example 3: Static Visualization")
    print("=" * 60)

    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    plotter = StaticGreekPlotter(K=100, r=0.05, sigma=0.2)

    # Create single Greek surface
    print("\nCreating static Gamma surface...")
    fig = plotter.plot_greek_surface(
        greek='gamma',
        option_type='call',
        current_S=100,
        current_T=0.5
    )
    plotter.save_figure(fig, 'gamma_surface.png')
    plt.close(fig)
    print("  Saved: gamma_surface.png")

    # Create all Greeks poster
    print("\nCreating Greek surfaces poster...")
    fig = plotter.plot_all_greeks(
        option_type='call',
        current_S=100,
        current_T=0.5,
        save_path='all_greeks_poster.png'
    )
    plt.close(fig)
    print("  Saved: all_greeks_poster.png")


def example_compare_call_put():
    """Compare call and put Greeks."""
    print("\n" + "=" * 60)
    print("Example 4: Call vs Put Comparison")
    print("=" * 60)

    bs = BlackScholes(S=100, K=100, T=0.5, r=0.05, sigma=0.2)

    print("\n{:<12} {:>12} {:>12}".format("Greek", "Call", "Put"))
    print("-" * 38)

    for greek in ['Delta', 'Theta', 'Rho']:
        call_val = getattr(bs, greek.lower())('call')
        put_val = getattr(bs, greek.lower())('put')
        print(f"{greek:<12} {call_val:>12.6f} {put_val:>12.6f}")

    # Gamma and Vega are the same
    print(f"{'Gamma':<12} {bs.gamma():>12.6f} {bs.gamma():>12.6f}")
    print(f"{'Vega':<12} {bs.vega():>12.6f} {bs.vega():>12.6f}")

    # Put-Call parity verification
    call = bs.call_price()
    put = bs.put_price()
    parity_diff = call - put - (bs.S - bs.K * np.exp(-bs.r * bs.T))
    print(f"\nPut-Call Parity: C - P = S - Ke^(-rT)")
    print(f"  Difference: {parity_diff:.2e} (should be ~0)")


def example_greeks_across_moneyness():
    """Show how Greeks vary across moneyness."""
    print("\n" + "=" * 60)
    print("Example 5: Greeks Across Moneyness")
    print("=" * 60)

    import numpy as np

    K = 100
    T = 0.5
    r = 0.05
    sigma = 0.2

    print("\nDelta at different moneyness levels (T=0.5, σ=20%):")
    print("{:<12} {:>12} {:>12}".format("S/K", "Call Delta", "Put Delta"))
    print("-" * 38)

    for moneyness in [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]:
        S = moneyness * K
        bs = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma)
        print(f"{moneyness:<12.2f} {bs.delta('call'):>12.4f} {bs.delta('put'):>12.4f}")


if __name__ == "__main__":
    import numpy as np

    print("\n" + "=" * 60)
    print("   BLACK-SCHOLES GREEK SURFACE - QUICK START EXAMPLES")
    print("=" * 60)

    # Run all examples
    example_pricing()
    example_compare_call_put()
    example_greeks_across_moneyness()

    try:
        example_interactive_visualization()
    except ImportError as e:
        print(f"\nSkipping interactive visualization (missing dependency): {e}")

    try:
        example_static_visualization()
    except ImportError as e:
        print(f"\nSkipping static visualization (missing dependency): {e}")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
