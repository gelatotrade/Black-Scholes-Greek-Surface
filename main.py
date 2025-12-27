#!/usr/bin/env python3
"""
Black-Scholes Greek Surface Visualization - Main Application

A comprehensive tool for visualizing Black-Scholes option Greeks as 3D surfaces.

Usage:
    python main.py --interactive
    python main.py --static --greek delta --save delta_surface.png
    python main.py --all-greeks --save all_greeks.html
    python main.py --poster --save poster.png

Author: Black-Scholes Greek Surface Contributors
License: MIT
"""

import argparse
import sys
from typing import Optional

import numpy as np


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Black-Scholes Greek Surface Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive 3D visualization (opens in browser)
  python main.py --interactive --greek delta

  # All Greeks in one interactive plot
  python main.py --interactive --all-greeks

  # Static publication-quality plot
  python main.py --static --greek gamma --save gamma.png

  # Create poster with all Greeks
  python main.py --poster --save greeks_poster.png

  # Animation of Greek vs volatility
  python main.py --animate --greek delta --vary sigma

  # Show Greeks for specific position
  python main.py --interactive --greek delta --spot 105 --time 0.5
        """
    )

    # Output type
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Create interactive Plotly visualization'
    )
    output_group.add_argument(
        '--static', '-s',
        action='store_true',
        help='Create static matplotlib visualization'
    )
    output_group.add_argument(
        '--poster',
        action='store_true',
        help='Create full poster visualization'
    )
    output_group.add_argument(
        '--animate',
        action='store_true',
        help='Create animation of Greek vs parameter'
    )
    output_group.add_argument(
        '--calculate', '-c',
        action='store_true',
        help='Just calculate and display Greek values'
    )

    # Greek selection
    parser.add_argument(
        '--greek', '-g',
        type=str,
        default='delta',
        choices=['delta', 'gamma', 'theta', 'vega', 'rho', 'price', 'vanna', 'volga', 'all'],
        help='Greek to visualize (default: delta)'
    )

    parser.add_argument(
        '--all-greeks',
        action='store_true',
        help='Show all main Greeks (Delta, Gamma, Theta, Vega) in one plot'
    )

    # Option parameters
    parser.add_argument(
        '--type', '-t',
        type=str,
        default='call',
        choices=['call', 'put'],
        help='Option type (default: call)'
    )

    parser.add_argument(
        '--strike', '-K',
        type=float,
        default=100.0,
        help='Strike price (default: 100)'
    )

    parser.add_argument(
        '--rate', '-r',
        type=float,
        default=0.05,
        help='Risk-free interest rate (default: 0.05)'
    )

    parser.add_argument(
        '--sigma',
        type=float,
        default=0.2,
        help='Volatility (default: 0.2)'
    )

    parser.add_argument(
        '--dividend', '-q',
        type=float,
        default=0.0,
        help='Dividend yield (default: 0)'
    )

    # Current position
    parser.add_argument(
        '--spot', '-S',
        type=float,
        default=None,
        help='Current spot price (for marking on surface)'
    )

    parser.add_argument(
        '--time', '-T',
        type=float,
        default=None,
        help='Current time to expiration (for marking on surface)'
    )

    # Animation parameter
    parser.add_argument(
        '--vary',
        type=str,
        default='sigma',
        choices=['sigma', 'r', 'K'],
        help='Parameter to vary in animation (default: sigma)'
    )

    # Output options
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Save output to file (html for interactive, png for static)'
    )

    parser.add_argument(
        '--no-atm',
        action='store_true',
        help='Hide ATM line'
    )

    parser.add_argument(
        '--no-peak',
        action='store_true',
        help='Hide peak zone marker'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for static images (default: 300)'
    )

    return parser.parse_args()


def display_banner():
    """Display application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║   ██████╗ ██╗      █████╗  ██████╗██╗  ██╗    ███████╗ ██████╗██╗  ██╗   ║
║   ██╔══██╗██║     ██╔══██╗██╔════╝██║ ██╔╝    ██╔════╝██╔════╝██║  ██║   ║
║   ██████╔╝██║     ███████║██║     █████╔╝     ███████╗██║     ███████║   ║
║   ██╔══██╗██║     ██╔══██║██║     ██╔═██╗     ╚════██║██║     ██╔══██║   ║
║   ██████╔╝███████╗██║  ██║╚██████╗██║  ██╗    ███████║╚██████╗██║  ██║   ║
║   ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝    ╚══════╝ ╚═════╝╚═╝  ╚═╝   ║
║                                                                          ║
║              G R E E K   S U R F A C E   V I S U A L I Z E R             ║
║                                                                          ║
║    Θ + ½σ²S²Γ + rSΔ = rV    |    d₁ = [ln(S/K) + (r+σ²/2)τ] / σ√τ       ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def calculate_and_display(args: argparse.Namespace) -> None:
    """Calculate and display Greek values."""
    from src.black_scholes import BlackScholes

    S = args.spot if args.spot else args.strike  # ATM if not specified
    T = args.time if args.time else 0.5

    bs = BlackScholes(
        S=S,
        K=args.strike,
        T=T,
        r=args.rate,
        sigma=args.sigma,
        q=args.dividend
    )

    print(f"\n{'='*60}")
    print(f"Black-Scholes Analysis - {args.type.capitalize()} Option")
    print(f"{'='*60}")
    print(f"\nParameters:")
    print(f"  Spot Price (S):      ${S:.2f}")
    print(f"  Strike Price (K):    ${args.strike:.2f}")
    print(f"  Time to Exp (T):     {T:.4f} years ({T*365:.1f} days)")
    print(f"  Risk-free Rate (r):  {args.rate:.2%}")
    print(f"  Volatility (σ):      {args.sigma:.2%}")
    print(f"  Dividend Yield (q):  {args.dividend:.2%}")
    print(f"  Moneyness (S/K):     {S/args.strike:.4f}")

    print(f"\n{'-'*60}")
    print(f"Intermediate Values:")
    print(f"  d₁ = {bs.d1:.6f}")
    print(f"  d₂ = {bs.d2:.6f}")

    print(f"\n{'-'*60}")
    print(f"Option Prices:")
    print(f"  Call Price:  ${bs.call_price():.4f}")
    print(f"  Put Price:   ${bs.put_price():.4f}")

    print(f"\n{'-'*60}")
    print(f"First-Order Greeks ({args.type.capitalize()}):")
    print(f"  Delta (Δ):   {bs.delta(args.type):.6f}")
    print(f"  Gamma (Γ):   {bs.gamma():.6f}")
    print(f"  Theta (Θ):   {bs.theta(args.type):.6f} (per year)")
    print(f"  Theta:       {bs.theta_per_day(args.type):.6f} (per day)")
    print(f"  Vega (V):    {bs.vega():.6f}")
    print(f"  Rho (ρ):     {bs.rho(args.type):.6f}")

    print(f"\n{'-'*60}")
    print(f"Second-Order Greeks:")
    print(f"  Vanna:       {bs.vanna():.6f}")
    print(f"  Volga:       {bs.volga():.6f}")
    print(f"  Charm:       {bs.charm(args.type):.6f}")
    print(f"  Speed:       {bs.speed():.6f}")
    print(f"  Zomma:       {bs.zomma():.6f}")
    print(f"  Color:       {bs.color():.6f}")

    # Verify BS PDE
    from src.black_scholes import verify_bs_pde
    lhs, rhs = verify_bs_pde(S, args.strike, T, args.rate, args.sigma, args.dividend)
    print(f"\n{'-'*60}")
    print(f"BS PDE Verification: Θ + ½σ²S²Γ + rSΔ = rV")
    print(f"  LHS: {lhs:.6f}")
    print(f"  RHS: {rhs:.6f}")
    print(f"  Error: {abs(lhs-rhs):.2e}")
    print(f"{'='*60}\n")


def create_interactive_visualization(args: argparse.Namespace) -> None:
    """Create interactive Plotly visualization."""
    from src.visualizations import GreekSurfaceVisualizer

    viz = GreekSurfaceVisualizer(
        K=args.strike,
        r=args.rate,
        sigma=args.sigma,
        q=args.dividend
    )

    print("\nCreating interactive visualization...")

    if args.all_greeks or args.greek == 'all':
        fig = viz.plot_all_greeks(
            option_type=args.type,
            current_S=args.spot,
            current_T=args.time
        )
        default_filename = f'all_greeks_{args.type}.html'
    else:
        fig = viz.plot_greek_surface(
            greek=args.greek,
            option_type=args.type,
            current_S=args.spot,
            current_T=args.time,
            show_atm_line=not args.no_atm,
            show_peak=not args.no_peak
        )
        default_filename = f'{args.greek}_{args.type}.html'

    if args.save:
        filename = args.save
    else:
        filename = default_filename

    viz.save_html(fig, filename)
    print(f"Saved interactive visualization to: {filename}")
    print("Open this file in your web browser to view the interactive 3D surface.")

    # Try to open in browser
    try:
        import webbrowser
        webbrowser.open(f'file://{filename}')
    except Exception:
        pass


def create_static_visualization(args: argparse.Namespace) -> None:
    """Create static matplotlib visualization."""
    from src.static_plots import StaticGreekPlotter
    import matplotlib.pyplot as plt

    plotter = StaticGreekPlotter(
        K=args.strike,
        r=args.rate,
        sigma=args.sigma,
        q=args.dividend
    )

    print("\nCreating static visualization...")

    if args.all_greeks or args.greek == 'all':
        fig = plotter.plot_all_greeks(
            option_type=args.type,
            current_S=args.spot,
            current_T=args.time
        )
        default_filename = f'all_greeks_{args.type}.png'
    else:
        fig = plotter.plot_greek_surface(
            greek=args.greek,
            option_type=args.type,
            current_S=args.spot,
            current_T=args.time,
            show_atm_line=not args.no_atm,
            show_peak=not args.no_peak
        )
        default_filename = f'{args.greek}_{args.type}.png'

    if args.save:
        filename = args.save
    else:
        filename = default_filename

    plotter.save_figure(fig, filename, dpi=args.dpi)
    print(f"Saved static visualization to: {filename}")
    plt.close()


def create_poster(args: argparse.Namespace) -> None:
    """Create poster visualization."""
    from src.static_plots import StaticGreekPlotter
    import matplotlib.pyplot as plt

    plotter = StaticGreekPlotter(
        K=args.strike,
        r=args.rate,
        sigma=args.sigma,
        q=args.dividend
    )

    print("\nCreating poster visualization...")

    S = args.spot if args.spot else args.strike
    T = args.time if args.time else 0.5

    filename = args.save if args.save else f'greeks_poster_{args.type}.png'

    fig = plotter.create_poster(
        option_type=args.type,
        current_S=S,
        current_T=T,
        save_path=filename
    )

    print(f"Saved poster to: {filename}")
    plt.close()


def create_animation(args: argparse.Namespace) -> None:
    """Create animated visualization."""
    from src.visualizations import GreekSurfaceVisualizer

    viz = GreekSurfaceVisualizer(
        K=args.strike,
        r=args.rate,
        sigma=args.sigma,
        q=args.dividend
    )

    print(f"\nCreating animation of {args.greek} vs {args.vary}...")

    fig = viz.create_animation(
        greek=args.greek,
        option_type=args.type,
        parameter=args.vary
    )

    filename = args.save if args.save else f'{args.greek}_vs_{args.vary}.html'
    viz.save_html(fig, filename)
    print(f"Saved animation to: {filename}")

    try:
        import webbrowser
        webbrowser.open(f'file://{filename}')
    except Exception:
        pass


def main():
    """Main entry point."""
    display_banner()
    args = parse_arguments()

    print(f"\nOption Type: {args.type.capitalize()}")
    print(f"Strike: ${args.strike:.2f}")
    print(f"Volatility: {args.sigma:.1%}")
    print(f"Rate: {args.rate:.2%}")

    if args.calculate:
        calculate_and_display(args)
    elif args.interactive:
        create_interactive_visualization(args)
    elif args.static:
        create_static_visualization(args)
    elif args.poster:
        create_poster(args)
    elif args.animate:
        create_animation(args)

    print("\nDone!")


if __name__ == "__main__":
    main()
