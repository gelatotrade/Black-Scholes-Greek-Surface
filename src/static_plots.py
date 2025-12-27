"""
Static Publication-Quality Greek Surface Plots using Matplotlib

Creates beautiful, publication-ready 3D surface plots with a dark theme
matching the style shown in financial publications.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from typing import Optional, Tuple, List, Dict
import matplotlib.patheffects as path_effects

from .black_scholes import BlackScholes


class StaticGreekPlotter:
    """
    Static Publication-Quality Greek Surface Plotter

    Creates beautiful matplotlib-based 3D surface plots with dark themes,
    custom colormaps, and professional styling suitable for publications.

    Parameters:
    -----------
    K : float
        Strike price (default: 100)
    r : float
        Risk-free interest rate (default: 0.05)
    sigma : float
        Volatility (default: 0.2)
    q : float
        Dividend yield (default: 0)

    Example:
    --------
    >>> plotter = StaticGreekPlotter(K=100, r=0.05, sigma=0.2)
    >>> fig = plotter.plot_all_greeks('call')
    >>> plt.show()
    """

    # Custom colormaps that look great on dark backgrounds
    CUSTOM_COLORMAPS = {
        'rainbow_dark': [
            (0.0, '#0000FF'),   # Blue
            (0.25, '#00FFFF'),  # Cyan
            (0.5, '#00FF00'),   # Green
            (0.75, '#FFFF00'),  # Yellow
            (1.0, '#FF0000')    # Red
        ],
        'thermal': [
            (0.0, '#000080'),   # Dark blue
            (0.25, '#0080FF'),  # Light blue
            (0.5, '#00FF80'),   # Cyan-green
            (0.75, '#FFFF00'),  # Yellow
            (1.0, '#FF0000')    # Red
        ]
    }

    GREEK_INFO = {
        'delta': {
            'name': 'Delta',
            'symbol': 'Δ',
            'call_formula': r'$\Delta_C = \Phi(d_1)$',
            'put_formula': r'$\Delta_P = \Phi(d_1) - 1$'
        },
        'gamma': {
            'name': 'Gamma',
            'symbol': 'Γ',
            'formula': r'$\Gamma = \frac{\phi(d_1)}{S \cdot \sigma \cdot \sqrt{\tau}}$'
        },
        'theta': {
            'name': 'Theta',
            'symbol': 'Θ',
            'call_formula': r'$\Theta_C = -\frac{S \cdot \sigma \cdot \phi(d_1)}{2\sqrt{\tau}} - rKe^{-r\tau}\Phi(d_2)$',
            'put_formula': r'$\Theta_P = -\frac{S \cdot \sigma \cdot \phi(d_1)}{2\sqrt{\tau}} + rKe^{-r\tau}\Phi(-d_2)$'
        },
        'vega': {
            'name': 'Vega',
            'symbol': 'V',
            'formula': r'$\mathcal{V} = S\sqrt{\tau} \cdot \phi(d_1)$'
        },
        'rho': {
            'name': 'Rho',
            'symbol': 'ρ',
            'call_formula': r'$\rho_C = K\tau e^{-r\tau}\Phi(d_2)$',
            'put_formula': r'$\rho_P = -K\tau e^{-r\tau}\Phi(-d_2)$'
        }
    }

    def __init__(
        self,
        K: float = 100,
        r: float = 0.05,
        sigma: float = 0.2,
        q: float = 0.0
    ):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.q = q

        # Default ranges
        self.moneyness_range = np.linspace(0.7, 1.3, 80)
        self.time_range = np.linspace(0.02, 2.0, 80)

        # Setup matplotlib style
        self._setup_style()

    def _setup_style(self):
        """Configure matplotlib for dark theme."""
        plt.style.use('dark_background')
        plt.rcParams.update({
            'figure.facecolor': '#0a0a0a',
            'axes.facecolor': '#0a0a0a',
            'axes.edgecolor': '#333333',
            'axes.labelcolor': 'white',
            'text.color': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white',
            'grid.color': '#333333',
            'font.family': 'sans-serif',
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 11,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })

    def _create_colormap(self, name: str = 'rainbow_dark') -> LinearSegmentedColormap:
        """Create a custom colormap."""
        if name in self.CUSTOM_COLORMAPS:
            colors = self.CUSTOM_COLORMAPS[name]
            positions = [c[0] for c in colors]
            hex_colors = [c[1] for c in colors]

            # Convert hex to RGB
            rgb_colors = []
            for hex_color in hex_colors:
                hex_color = hex_color.lstrip('#')
                rgb = tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))
                rgb_colors.append(rgb)

            return LinearSegmentedColormap.from_list(name, list(zip(positions, rgb_colors)))
        return plt.cm.get_cmap(name)

    def _calculate_surface(
        self,
        greek: str,
        option_type: str = 'call'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Greek surface values."""
        S_range = self.moneyness_range * self.K
        surface = BlackScholes.calculate_surface(
            S_range=S_range,
            T_range=self.time_range,
            K=self.K,
            r=self.r,
            sigma=self.sigma,
            greek=greek,
            option_type=option_type,
            q=self.q
        )
        return self.moneyness_range, self.time_range, surface

    def _find_peak_zone(
        self,
        surface: np.ndarray,
        moneyness: np.ndarray,
        time: np.ndarray
    ) -> Tuple[float, float, float]:
        """Find the peak value location on the surface."""
        abs_surface = np.abs(surface)
        max_idx = np.unravel_index(np.argmax(abs_surface), abs_surface.shape)
        return (
            moneyness[max_idx[1]],
            time[max_idx[0]],
            surface[max_idx]
        )

    def plot_greek_surface(
        self,
        greek: str = 'delta',
        option_type: str = 'call',
        current_S: Optional[float] = None,
        current_T: Optional[float] = None,
        show_atm_line: bool = True,
        show_peak: bool = True,
        figsize: Tuple[int, int] = (10, 8),
        colormap: str = 'rainbow_dark',
        elevation: float = 25,
        azimuth: float = -45
    ) -> plt.Figure:
        """
        Create a single Greek surface plot.

        Parameters:
        -----------
        greek : str
            Greek to plot ('delta', 'gamma', 'theta', 'vega', 'rho')
        option_type : str
            'call' or 'put'
        current_S : float, optional
            Current spot price
        current_T : float, optional
            Current time to expiration
        show_atm_line : bool
            Whether to show ATM line
        show_peak : bool
            Whether to show peak marker
        figsize : tuple
            Figure size
        colormap : str
            Colormap name
        elevation : float
            View elevation angle
        azimuth : float
            View azimuth angle

        Returns:
        --------
        plt.Figure : Matplotlib figure
        """
        # Calculate surface
        moneyness, time, surface = self._calculate_surface(greek, option_type)
        M, T = np.meshgrid(moneyness, time)

        # Create figure
        fig = plt.figure(figsize=figsize, facecolor='#0a0a0a')
        ax = fig.add_subplot(111, projection='3d')

        # Get colormap
        if colormap in self.CUSTOM_COLORMAPS:
            cmap = self._create_colormap(colormap)
        else:
            cmap = plt.cm.get_cmap(colormap)

        # Plot surface
        surf = ax.plot_surface(
            M, T, surface,
            cmap=cmap,
            edgecolor='none',
            alpha=0.9,
            antialiased=True,
            rcount=80,
            ccount=80
        )

        # Add ATM line
        if show_atm_line:
            atm_idx = np.argmin(np.abs(moneyness - 1.0))
            atm_values = surface[:, atm_idx]
            ax.plot(
                [1.0] * len(time), time, atm_values,
                color='lime', linewidth=3, label='ATM',
                zorder=10
            )

        # Add current position marker
        if current_S is not None and current_T is not None:
            bs = BlackScholes(
                S=current_S, K=self.K, T=current_T,
                r=self.r, sigma=self.sigma, q=self.q
            )
            greek_methods = {
                'delta': lambda: bs.delta(option_type),
                'gamma': lambda: bs.gamma(),
                'theta': lambda: bs.theta(option_type),
                'vega': lambda: bs.vega(),
                'rho': lambda: bs.rho(option_type)
            }
            value = greek_methods[greek]()
            ax.scatter(
                [current_S / self.K], [current_T], [value],
                color='red', s=100, marker='o',
                label=f'Current S/K', zorder=15
            )

        # Add peak marker
        if show_peak:
            peak_m, peak_t, peak_val = self._find_peak_zone(surface, moneyness, time)
            ax.scatter(
                [peak_m], [peak_t], [peak_val],
                color='yellow', s=150, marker='D',
                label='Peak Zone', zorder=15
            )

        # Get Greek info
        info = self.GREEK_INFO.get(greek, {})
        greek_name = info.get('name', greek.capitalize())
        greek_symbol = info.get('symbol', greek[0].upper())

        # Get formula
        formula = info.get('formula', '')
        if not formula:
            if option_type.lower() == 'call':
                formula = info.get('call_formula', '')
            else:
                formula = info.get('put_formula', '')

        # Set labels and title
        ax.set_xlabel('S/K', fontsize=12, labelpad=10)
        ax.set_ylabel('T (Years)', fontsize=12, labelpad=10)
        ax.set_zlabel(greek_symbol, fontsize=12, labelpad=10)

        title = f'{greek_name}\n{formula}'
        ax.set_title(title, fontsize=14, pad=20, color='white')

        # Set view angle
        ax.view_init(elev=elevation, azim=azimuth)

        # Style the axes
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#333333')
        ax.yaxis.pane.set_edgecolor('#333333')
        ax.zaxis.pane.set_edgecolor('#333333')
        ax.grid(True, alpha=0.3)

        # Add legend
        ax.legend(loc='upper left', framealpha=0.7)

        # Add colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
        cbar.set_label(greek_symbol, fontsize=11)

        plt.tight_layout()
        return fig

    def plot_all_greeks(
        self,
        option_type: str = 'call',
        current_S: Optional[float] = None,
        current_T: Optional[float] = None,
        figsize: Tuple[int, int] = (16, 14),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a 2x2 subplot with Delta, Vega, Gamma, and Theta surfaces.

        This matches the style shown in the reference image with:
        - Dark background
        - Rainbow colormap
        - ATM lines in green
        - Current position markers
        - Peak zone markers
        - Mathematical formulas

        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        current_S : float, optional
            Current spot price
        current_T : float, optional
            Current time to expiration
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure

        Returns:
        --------
        plt.Figure : Matplotlib figure
        """
        greeks = ['delta', 'vega', 'gamma', 'theta']

        fig = plt.figure(figsize=figsize, facecolor='#0a0a0a')

        # Main title with BS PDE
        main_title = (
            r"$\bf{POV:\ Black\text{-}Scholes\ Greek\ Surfaces}$" + "\n"
            r"$\Theta + \frac{1}{2}\tilde{\sigma}^2 S^2 \Gamma + rS\Delta = rV$"
            r"$\qquad q(K) = e^{r\tilde{\tau}} \frac{\partial^2 C}{\partial K^2}$"
        )
        fig.suptitle(main_title, fontsize=16, color='white', y=0.98)

        # Create subplots
        for idx, greek in enumerate(greeks):
            ax = fig.add_subplot(2, 2, idx + 1, projection='3d')

            # Calculate surface
            moneyness, time, surface = self._calculate_surface(greek, option_type)
            M, T = np.meshgrid(moneyness, time)

            # Create colormap
            cmap = self._create_colormap('rainbow_dark')

            # Plot surface
            surf = ax.plot_surface(
                M, T, surface,
                cmap=cmap,
                edgecolor='none',
                alpha=0.9,
                antialiased=True,
                rcount=60,
                ccount=60
            )

            # Add ATM line
            atm_idx = np.argmin(np.abs(moneyness - 1.0))
            atm_values = surface[:, atm_idx]
            ax.plot(
                [1.0] * len(time), time, atm_values,
                color='lime', linewidth=3,
                zorder=10
            )

            # Add current position marker
            if current_S is not None and current_T is not None:
                bs = BlackScholes(
                    S=current_S, K=self.K, T=current_T,
                    r=self.r, sigma=self.sigma, q=self.q
                )
                greek_methods = {
                    'delta': lambda bs=bs: bs.delta(option_type),
                    'gamma': lambda bs=bs: bs.gamma(),
                    'theta': lambda bs=bs: bs.theta(option_type),
                    'vega': lambda bs=bs: bs.vega()
                }
                value = greek_methods[greek]()
                ax.scatter(
                    [current_S / self.K], [current_T], [value],
                    color='red', s=80, marker='o',
                    zorder=15
                )

            # Add peak marker
            peak_m, peak_t, peak_val = self._find_peak_zone(surface, moneyness, time)
            ax.scatter(
                [peak_m], [peak_t], [peak_val],
                color='yellow', s=100, marker='D',
                zorder=15
            )

            # Get Greek info
            info = self.GREEK_INFO.get(greek, {})
            greek_name = info.get('name', greek.capitalize())
            greek_symbol = info.get('symbol', greek[0].upper())

            # Get formula
            formula = info.get('formula', '')
            if not formula:
                if option_type.lower() == 'call':
                    formula = info.get('call_formula', '')
                else:
                    formula = info.get('put_formula', '')

            # Set title with formula
            title_color = {
                'delta': '#FF69B4',   # Pink
                'vega': '#FF69B4',    # Pink
                'gamma': '#FF69B4',   # Pink
                'theta': '#FF69B4'    # Pink
            }
            ax.set_title(
                f'{greek_name}\n{formula}',
                fontsize=11,
                color=title_color.get(greek, 'white'),
                pad=5
            )

            # Set labels
            ax.set_xlabel('S/K', fontsize=10, labelpad=8)
            ax.set_ylabel('T (Years)', fontsize=10, labelpad=8)
            ax.set_zlabel('', fontsize=10, labelpad=5)

            # Set view angle (different for each Greek for better visualization)
            view_angles = {
                'delta': (25, -45),
                'vega': (25, -135),
                'gamma': (25, -45),
                'theta': (25, -135)
            }
            elev, azim = view_angles.get(greek, (25, -45))
            ax.view_init(elev=elev, azim=azim)

            # Style axes
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('#333333')
            ax.yaxis.pane.set_edgecolor('#333333')
            ax.zaxis.pane.set_edgecolor('#333333')
            ax.grid(True, alpha=0.2)

            # Tick styling
            ax.tick_params(axis='both', which='major', labelsize=8)

        # Add legend at bottom
        legend_elements = [
            plt.Line2D([0], [0], color='lime', linewidth=3, label='ATM'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                       markersize=10, label='Current S/K', linestyle='None'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='yellow',
                       markersize=10, label='Peak Zone', linestyle='None')
        ]

        fig.legend(
            handles=legend_elements,
            loc='lower center',
            ncol=3,
            fontsize=11,
            framealpha=0.7,
            bbox_to_anchor=(0.5, 0.02)
        )

        # Add formulas at bottom
        formula_text = (
            r"$\rho_C = K\tilde{\tau}e^{-r\tilde{\tau}}\Phi(d_2)$"
            r"$\qquad\qquad$"
            r"$\Theta_C = -\frac{S\phi(d_1)\tilde{\sigma}}{2\sqrt{\tilde{\tau}}} - rKe^{-r\tilde{\tau}}\Phi(d_2)$"
            "\n"
            r"$\tilde{\tau} = \max(\tau, \varepsilon_\tau),\quad$"
            r"$\tilde{\sigma} = \max(\sigma, \varepsilon_\sigma),\quad$"
            r"$\varepsilon_\tau, \varepsilon_\sigma > 0$"
            "\n"
            r"$d_1 = \frac{\ln(S/K) + (r + \frac{1}{2}\tilde{\sigma}^2)\tilde{\tau}}{\tilde{\sigma}\sqrt{\tilde{\tau}}}$"
            r"$\qquad$"
            r"$d_2 = d_1 - \tilde{\sigma}\sqrt{\tilde{\tau}}$"
        )

        fig.text(
            0.5, 0.08, formula_text,
            ha='center', va='top',
            fontsize=10, color='white',
            transform=fig.transFigure
        )

        plt.tight_layout(rect=[0, 0.15, 1, 0.95])

        if save_path:
            plt.savefig(
                save_path,
                dpi=300,
                facecolor='#0a0a0a',
                edgecolor='none',
                bbox_inches='tight'
            )

        return fig

    def plot_greek_vs_parameter(
        self,
        greek: str = 'delta',
        parameter: str = 'S',
        option_type: str = 'call',
        param_range: Optional[np.ndarray] = None,
        T_values: List[float] = [0.1, 0.5, 1.0, 2.0],
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Create a 2D plot showing how a Greek varies with a parameter
        for different times to expiration.

        Parameters:
        -----------
        greek : str
            Greek to plot
        parameter : str
            Parameter to vary ('S', 'sigma', 'r')
        option_type : str
            'call' or 'put'
        param_range : np.ndarray, optional
            Range of parameter values
        T_values : list
            Times to expiration for different curves
        figsize : tuple
            Figure size

        Returns:
        --------
        plt.Figure : Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize, facecolor='#0a0a0a')
        ax.set_facecolor('#0a0a0a')

        # Default parameter ranges
        if param_range is None:
            if parameter == 'S':
                param_range = np.linspace(50, 150, 100)
            elif parameter == 'sigma':
                param_range = np.linspace(0.05, 0.6, 100)
            elif parameter == 'r':
                param_range = np.linspace(0.001, 0.15, 100)

        # Color gradient for different T values
        colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(T_values)))

        greek_methods = {
            'delta': lambda bs: bs.delta(option_type),
            'gamma': lambda bs: bs.gamma(),
            'theta': lambda bs: bs.theta(option_type),
            'vega': lambda bs: bs.vega(),
            'rho': lambda bs: bs.rho(option_type)
        }

        for T, color in zip(T_values, colors):
            values = []
            for param_val in param_range:
                if parameter == 'S':
                    bs = BlackScholes(param_val, self.K, T, self.r, self.sigma, self.q)
                elif parameter == 'sigma':
                    bs = BlackScholes(self.K, self.K, T, self.r, param_val, self.q)
                elif parameter == 'r':
                    bs = BlackScholes(self.K, self.K, T, param_val, self.sigma, self.q)

                values.append(greek_methods[greek](bs))

            ax.plot(param_range, values, color=color, linewidth=2, label=f'T = {T}')

        # Get Greek info
        info = self.GREEK_INFO.get(greek, {})
        greek_name = info.get('name', greek.capitalize())
        greek_symbol = info.get('symbol', greek[0].upper())

        # Labels and title
        param_labels = {'S': 'Spot Price (S)', 'sigma': 'Volatility (σ)', 'r': 'Interest Rate (r)'}
        ax.set_xlabel(param_labels.get(parameter, parameter), fontsize=12)
        ax.set_ylabel(greek_symbol, fontsize=12)
        ax.set_title(f'{greek_name} vs {param_labels.get(parameter, parameter)}', fontsize=14)

        # Add ATM line if showing vs S
        if parameter == 'S':
            ax.axvline(x=self.K, color='lime', linestyle='--', linewidth=1, alpha=0.7, label='ATM')

        ax.legend(loc='best', framealpha=0.7)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_pnl_surface(
        self,
        option_type: str = 'call',
        position: str = 'long',
        premium_paid: Optional[float] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Create a P&L surface showing profit/loss at expiration.

        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        position : str
            'long' or 'short'
        premium_paid : float, optional
            Premium paid (if None, calculated using BS)
        figsize : tuple
            Figure size

        Returns:
        --------
        plt.Figure : Matplotlib figure
        """
        # Calculate current option price as premium
        if premium_paid is None:
            bs = BlackScholes(self.K, self.K, 0.5, self.r, self.sigma, self.q)
            premium_paid = bs.price(option_type)

        fig = plt.figure(figsize=figsize, facecolor='#0a0a0a')
        ax = fig.add_subplot(111, projection='3d')

        # Create meshgrid
        S_final = np.linspace(0.5 * self.K, 1.5 * self.K, 100)
        T_remaining = np.linspace(0.01, 1.0, 50)
        S_mesh, T_mesh = np.meshgrid(S_final, T_remaining)

        # Calculate P&L
        pnl = np.zeros_like(S_mesh)
        for i in range(len(T_remaining)):
            for j in range(len(S_final)):
                bs = BlackScholes(S_final[j], self.K, T_remaining[i], self.r, self.sigma, self.q)
                current_value = bs.price(option_type)

                if position == 'long':
                    pnl[i, j] = current_value - premium_paid
                else:
                    pnl[i, j] = premium_paid - current_value

        # Create custom colormap (green for profit, red for loss)
        colors = ['#FF0000', '#FF6666', '#FFFFFF', '#66FF66', '#00FF00']
        cmap = LinearSegmentedColormap.from_list('pnl', colors)

        # Normalize around zero
        vmax = max(abs(pnl.min()), abs(pnl.max()))
        norm = Normalize(vmin=-vmax, vmax=vmax)

        # Plot surface
        surf = ax.plot_surface(
            S_mesh / self.K, T_mesh, pnl,
            cmap=cmap,
            norm=norm,
            edgecolor='none',
            alpha=0.9,
            antialiased=True
        )

        # Add breakeven line (P&L = 0)
        ax.contour(
            S_mesh / self.K, T_mesh, pnl,
            levels=[0],
            colors=['white'],
            linewidths=2
        )

        # Labels and title
        ax.set_xlabel('S/K', fontsize=12, labelpad=10)
        ax.set_ylabel('T (Years)', fontsize=12, labelpad=10)
        ax.set_zlabel('P&L', fontsize=12, labelpad=10)

        title = f'{position.capitalize()} {option_type.capitalize()} P&L Surface'
        ax.set_title(title, fontsize=14, pad=20)

        ax.view_init(elev=25, azim=-45)

        # Style axes
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
        cbar.set_label('P&L', fontsize=11)

        plt.tight_layout()
        return fig

    def create_poster(
        self,
        option_type: str = 'call',
        current_S: float = 100,
        current_T: float = 0.5,
        save_path: str = 'greeks_poster.png'
    ) -> plt.Figure:
        """
        Create a complete poster-style visualization with all Greeks and formulas.

        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        current_S : float
            Current spot price
        current_T : float
            Current time to expiration
        save_path : str
            Path to save the poster

        Returns:
        --------
        plt.Figure : Matplotlib figure
        """
        fig = plt.figure(figsize=(20, 24), facecolor='#0a0a0a')

        # Title
        fig.suptitle(
            'POV: Black-Scholes Greek Surfaces',
            fontsize=28,
            fontweight='bold',
            color='white',
            y=0.97
        )

        # BS PDE formula
        pde_text = (
            r'$\Theta + \frac{1}{2}\sigma^2 S^2 \Gamma + rS\Delta = rV$'
            r'$\qquad$'
            r'$q(K) = e^{r\tau} \frac{\partial^2 C}{\partial K^2}$'
        )
        fig.text(0.5, 0.94, pde_text, ha='center', fontsize=16, color='white')

        # Create grid for all Greeks
        greeks = ['delta', 'vega', 'gamma', 'theta', 'rho', 'price']

        for idx, greek in enumerate(greeks):
            row = idx // 2
            col = idx % 2
            ax = fig.add_subplot(3, 2, idx + 1, projection='3d')

            # Calculate surface
            moneyness, time, surface = self._calculate_surface(greek, option_type)
            M, T = np.meshgrid(moneyness, time)

            # Plot
            cmap = self._create_colormap('rainbow_dark')
            surf = ax.plot_surface(
                M, T, surface,
                cmap=cmap,
                edgecolor='none',
                alpha=0.9,
                antialiased=True,
                rcount=60,
                ccount=60
            )

            # ATM line
            atm_idx = np.argmin(np.abs(moneyness - 1.0))
            ax.plot(
                [1.0] * len(time), time, surface[:, atm_idx],
                color='lime', linewidth=3
            )

            # Current position
            bs = BlackScholes(current_S, self.K, current_T, self.r, self.sigma, self.q)
            greek_methods = {
                'delta': lambda bs=bs: bs.delta(option_type),
                'gamma': lambda bs=bs: bs.gamma(),
                'theta': lambda bs=bs: bs.theta(option_type),
                'vega': lambda bs=bs: bs.vega(),
                'rho': lambda bs=bs: bs.rho(option_type),
                'price': lambda bs=bs: bs.price(option_type)
            }
            value = greek_methods[greek]()
            ax.scatter([current_S / self.K], [current_T], [value],
                      color='red', s=80, marker='o')

            # Peak
            peak_m, peak_t, peak_val = self._find_peak_zone(surface, moneyness, time)
            ax.scatter([peak_m], [peak_t], [peak_val],
                      color='yellow', s=100, marker='D')

            # Title
            info = self.GREEK_INFO.get(greek, {'name': greek.capitalize(), 'symbol': greek[0].upper()})
            ax.set_title(
                f"{info['name']} ({info.get('symbol', '')})",
                fontsize=14,
                color='#FF69B4',
                pad=10
            )

            ax.set_xlabel('S/K', fontsize=10)
            ax.set_ylabel('T (Years)', fontsize=10)

            ax.view_init(elev=25, azim=-45 if col == 0 else -135)

            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

        # Legend
        legend_elements = [
            plt.Line2D([0], [0], color='lime', linewidth=3, label='ATM'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                       markersize=10, label='Current S/K', linestyle='None'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='yellow',
                       markersize=10, label='Peak Zone', linestyle='None')
        ]

        fig.legend(
            handles=legend_elements,
            loc='lower center',
            ncol=3,
            fontsize=12,
            framealpha=0.7,
            bbox_to_anchor=(0.5, 0.02)
        )

        # Formulas
        formulas = (
            r'$\Delta_C = \Phi(d_1),\quad \Delta_P = \Phi(d_1) - 1$' + '\n'
            r'$\Gamma = \frac{\phi(d_1)}{S\sigma\sqrt{\tau}}$' + '\n'
            r'$\mathcal{V} = S\sqrt{\tau}\phi(d_1)$' + '\n'
            r'$\rho_C = K\tau e^{-r\tau}\Phi(d_2)$' + '\n'
            r'$d_1 = \frac{\ln(S/K) + (r + \frac{1}{2}\sigma^2)\tau}{\sigma\sqrt{\tau}},\quad$'
            r'$d_2 = d_1 - \sigma\sqrt{\tau}$'
        )

        fig.text(0.5, 0.06, formulas, ha='center', fontsize=12, color='white')

        plt.tight_layout(rect=[0, 0.1, 1, 0.93])

        if save_path:
            plt.savefig(
                save_path,
                dpi=300,
                facecolor='#0a0a0a',
                edgecolor='none',
                bbox_inches='tight'
            )

        return fig

    def save_figure(
        self,
        fig: plt.Figure,
        path: str,
        dpi: int = 300,
        format: str = 'png'
    ) -> None:
        """Save figure to file."""
        fig.savefig(
            path,
            dpi=dpi,
            format=format,
            facecolor='#0a0a0a',
            edgecolor='none',
            bbox_inches='tight'
        )


if __name__ == "__main__":
    print("Creating static Greek surface plots...")

    plotter = StaticGreekPlotter(K=100, r=0.05, sigma=0.2)

    # Create single Greek plot
    fig = plotter.plot_greek_surface(
        greek='delta',
        option_type='call',
        current_S=100,
        current_T=0.5
    )
    plotter.save_figure(fig, 'delta_surface.png')
    plt.close()

    # Create all Greeks
    fig = plotter.plot_all_greeks(
        option_type='call',
        current_S=100,
        current_T=0.5,
        save_path='all_greeks_surface.png'
    )
    plt.close()

    # Create poster
    fig = plotter.create_poster(
        option_type='call',
        current_S=100,
        current_T=0.5,
        save_path='greeks_poster.png'
    )
    plt.close()

    print("Done! Check the generated PNG files.")
