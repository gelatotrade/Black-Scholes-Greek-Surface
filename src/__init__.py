"""
Black-Scholes Greek Surface Visualization Package

A comprehensive library for calculating and visualizing Black-Scholes
option pricing Greeks as 3D surfaces.
"""

from .black_scholes import BlackScholes
from .visualizations import GreekSurfaceVisualizer
from .static_plots import StaticGreekPlotter

__version__ = "1.0.0"
__author__ = "Black-Scholes Greek Surface Contributors"
__all__ = ["BlackScholes", "GreekSurfaceVisualizer", "StaticGreekPlotter"]
