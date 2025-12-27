"""
Interactive 3D Greek Surface Visualizations using Plotly

Creates beautiful, interactive 3D surface plots for Black-Scholes Greeks
that can be viewed in a web browser with zoom, rotate, and hover capabilities.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Tuple, Dict, Any
import plotly.io as pio

from .black_scholes import BlackScholes


class GreekSurfaceVisualizer:
    """
    Interactive 3D Greek Surface Visualization using Plotly

    Creates stunning visualizations of Black-Scholes Greeks as 3D surfaces
    with interactive features like rotation, zoom, and hover information.

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
    >>> viz = GreekSurfaceVisualizer(K=100, r=0.05, sigma=0.2)
    >>> fig = viz.plot_greek_surface('delta', 'call')
    >>> fig.show()
    """

    # Color schemes for different Greeks
    COLORSCALES = {
        'delta': 'RdYlBu',
        'gamma': 'Viridis',
        'theta': 'RdBu_r',
        'vega': 'Plasma',
        'rho': 'Cividis',
        'price': 'Turbo',
        'vanna': 'Picnic',
        'volga': 'Hot'
    }

    # Greek display names and formulas
    GREEK_INFO = {
        'delta': {
            'name': 'Delta (Δ)',
            'call_formula': 'Δ_C = Φ(d₁)',
            'put_formula': 'Δ_P = Φ(d₁) - 1',
            'description': 'Rate of change of option value with respect to underlying price'
        },
        'gamma': {
            'name': 'Gamma (Γ)',
            'formula': 'Γ = φ(d₁) / (S·σ·√τ)',
            'description': 'Rate of change of delta with respect to underlying price'
        },
        'theta': {
            'name': 'Theta (Θ)',
            'call_formula': 'Θ_C = -S·σ·φ(d₁)/(2√τ) - r·K·e^(-rτ)·Φ(d₂)',
            'put_formula': 'Θ_P = -S·σ·φ(d₁)/(2√τ) + r·K·e^(-rτ)·Φ(-d₂)',
            'description': 'Rate of decline in option value due to time passage'
        },
        'vega': {
            'name': 'Vega (V)',
            'formula': 'V = S·√τ·φ(d₁)',
            'description': 'Rate of change of option value with respect to volatility'
        },
        'rho': {
            'name': 'Rho (ρ)',
            'call_formula': 'ρ_C = K·τ·e^(-rτ)·Φ(d₂)',
            'put_formula': 'ρ_P = -K·τ·e^(-rτ)·Φ(-d₂)',
            'description': 'Rate of change of option value with respect to interest rate'
        },
        'price': {
            'name': 'Option Price',
            'call_formula': 'C = S·Φ(d₁) - K·e^(-rτ)·Φ(d₂)',
            'put_formula': 'P = K·e^(-rτ)·Φ(-d₂) - S·Φ(-d₁)',
            'description': 'Black-Scholes option price'
        },
        'vanna': {
            'name': 'Vanna',
            'formula': 'Vanna = -φ(d₁)·d₂/σ',
            'description': 'Sensitivity of delta to volatility (or vega to spot)'
        },
        'volga': {
            'name': 'Volga (Vomma)',
            'formula': 'Volga = Vega·d₁·d₂/σ',
            'description': 'Sensitivity of vega to volatility'
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

        # Default ranges (using moneyness S/K)
        self.moneyness_range = np.linspace(0.7, 1.3, 60)  # S/K from 0.7 to 1.3
        self.time_range = np.linspace(0.02, 2.0, 60)  # 1 week to 2 years

    def _calculate_surface(
        self,
        greek: str,
        option_type: str = 'call'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Greek surface values."""
        S_range = self.moneyness_range * self.K  # Convert moneyness to actual S
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

    def _add_atm_line(
        self,
        fig: go.Figure,
        surface: np.ndarray,
        time: np.ndarray
    ) -> go.Figure:
        """Add At-The-Money (S/K = 1) line on the surface."""
        # Find ATM index
        atm_idx = np.argmin(np.abs(self.moneyness_range - 1.0))
        atm_values = surface[:, atm_idx]

        fig.add_trace(go.Scatter3d(
            x=[1.0] * len(time),
            y=time,
            z=atm_values,
            mode='lines',
            line=dict(color='lime', width=6),
            name='ATM (S/K = 1)'
        ))

        return fig

    def _add_current_point(
        self,
        fig: go.Figure,
        current_S: float,
        current_T: float,
        greek: str,
        option_type: str
    ) -> go.Figure:
        """Add current position marker."""
        bs = BlackScholes(
            S=current_S, K=self.K, T=current_T,
            r=self.r, sigma=self.sigma, q=self.q
        )

        greek_methods = {
            'delta': lambda: bs.delta(option_type),
            'gamma': lambda: bs.gamma(),
            'theta': lambda: bs.theta(option_type),
            'vega': lambda: bs.vega(),
            'rho': lambda: bs.rho(option_type),
            'price': lambda: bs.price(option_type),
            'vanna': lambda: bs.vanna(),
            'volga': lambda: bs.volga()
        }

        value = greek_methods[greek]()

        fig.add_trace(go.Scatter3d(
            x=[current_S / self.K],
            y=[current_T],
            z=[value],
            mode='markers',
            marker=dict(size=10, color='red', symbol='circle'),
            name=f'Current S/K ({current_S/self.K:.2f})'
        ))

        return fig

    def _add_peak_marker(
        self,
        fig: go.Figure,
        surface: np.ndarray
    ) -> go.Figure:
        """Add peak zone marker."""
        peak_m, peak_t, peak_val = self._find_peak_zone(
            surface, self.moneyness_range, self.time_range
        )

        fig.add_trace(go.Scatter3d(
            x=[peak_m],
            y=[peak_t],
            z=[peak_val],
            mode='markers',
            marker=dict(size=12, color='yellow', symbol='diamond'),
            name='Peak Zone'
        ))

        return fig

    def plot_greek_surface(
        self,
        greek: str = 'delta',
        option_type: str = 'call',
        current_S: Optional[float] = None,
        current_T: Optional[float] = None,
        show_atm_line: bool = True,
        show_peak: bool = True,
        title: Optional[str] = None,
        colorscale: Optional[str] = None
    ) -> go.Figure:
        """
        Create an interactive 3D Greek surface plot.

        Parameters:
        -----------
        greek : str
            Greek to plot ('delta', 'gamma', 'theta', 'vega', 'rho', 'price', 'vanna', 'volga')
        option_type : str
            'call' or 'put'
        current_S : float, optional
            Current spot price to mark on surface
        current_T : float, optional
            Current time to expiration to mark on surface
        show_atm_line : bool
            Whether to show the ATM line
        show_peak : bool
            Whether to show peak zone marker
        title : str, optional
            Custom title
        colorscale : str, optional
            Plotly colorscale name

        Returns:
        --------
        go.Figure : Plotly figure object
        """
        # Calculate surface
        moneyness, time, surface = self._calculate_surface(greek, option_type)

        # Create meshgrid for plotting
        M, T = np.meshgrid(moneyness, time)

        # Get colorscale
        if colorscale is None:
            colorscale = self.COLORSCALES.get(greek, 'Viridis')

        # Get Greek info
        info = self.GREEK_INFO.get(greek, {})
        greek_name = info.get('name', greek.capitalize())

        # Create figure
        fig = go.Figure()

        # Add surface
        fig.add_trace(go.Surface(
            x=M,
            y=T,
            z=surface,
            colorscale=colorscale,
            name=greek_name,
            showscale=True,
            colorbar=dict(
                title=dict(text=greek_name, side='right'),
                thickness=20,
                len=0.75
            ),
            hovertemplate=(
                f'{greek_name}<br>'
                'S/K: %{x:.3f}<br>'
                'T (Years): %{y:.3f}<br>'
                'Value: %{z:.4f}<extra></extra>'
            ),
            opacity=0.9,
            contours=dict(
                z=dict(
                    show=True,
                    usecolormap=True,
                    highlightcolor="white",
                    project_z=True
                )
            )
        ))

        # Add ATM line
        if show_atm_line:
            fig = self._add_atm_line(fig, surface, time)

        # Add current position
        if current_S is not None and current_T is not None:
            fig = self._add_current_point(fig, current_S, current_T, greek, option_type)

        # Add peak marker
        if show_peak:
            fig = self._add_peak_marker(fig, surface)

        # Get formula
        formula = info.get('formula', '')
        if not formula:
            if option_type.lower() == 'call':
                formula = info.get('call_formula', '')
            else:
                formula = info.get('put_formula', '')

        # Create title
        if title is None:
            title = f"{greek_name} Surface - {option_type.capitalize()} Option"

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b><br><sup>{formula}</sup>",
                x=0.5,
                xanchor='center',
                font=dict(size=20)
            ),
            scene=dict(
                xaxis=dict(
                    title='S/K (Moneyness)',
                    backgroundcolor='rgb(20, 20, 20)',
                    gridcolor='rgb(50, 50, 50)',
                    showbackground=True,
                    tickformat='.2f'
                ),
                yaxis=dict(
                    title='T (Years)',
                    backgroundcolor='rgb(20, 20, 20)',
                    gridcolor='rgb(50, 50, 50)',
                    showbackground=True,
                    tickformat='.2f'
                ),
                zaxis=dict(
                    title=greek_name,
                    backgroundcolor='rgb(20, 20, 20)',
                    gridcolor='rgb(50, 50, 50)',
                    showbackground=True,
                    tickformat='.4f'
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                ),
                aspectratio=dict(x=1, y=1, z=0.8)
            ),
            paper_bgcolor='rgb(10, 10, 10)',
            plot_bgcolor='rgb(10, 10, 10)',
            font=dict(color='white'),
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='rgba(255,255,255,0.3)',
                borderwidth=1
            ),
            margin=dict(l=0, r=0, t=80, b=0)
        )

        return fig

    def plot_all_greeks(
        self,
        option_type: str = 'call',
        current_S: Optional[float] = None,
        current_T: Optional[float] = None
    ) -> go.Figure:
        """
        Create a 2x2 subplot with Delta, Vega, Gamma, and Theta surfaces.

        Parameters:
        -----------
        option_type : str
            'call' or 'put'
        current_S : float, optional
            Current spot price to mark
        current_T : float, optional
            Current time to expiration to mark

        Returns:
        --------
        go.Figure : Plotly figure with 4 subplots
        """
        greeks = ['delta', 'vega', 'gamma', 'theta']

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}],
                   [{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=[
                self.GREEK_INFO[g]['name'] for g in greeks
            ],
            horizontal_spacing=0.05,
            vertical_spacing=0.1
        )

        for idx, greek in enumerate(greeks):
            row = idx // 2 + 1
            col = idx % 2 + 1

            moneyness, time, surface = self._calculate_surface(greek, option_type)
            M, T = np.meshgrid(moneyness, time)

            colorscale = self.COLORSCALES.get(greek, 'Viridis')
            info = self.GREEK_INFO.get(greek, {})
            greek_name = info.get('name', greek.capitalize())

            fig.add_trace(
                go.Surface(
                    x=M,
                    y=T,
                    z=surface,
                    colorscale=colorscale,
                    showscale=False,
                    opacity=0.9,
                    hovertemplate=(
                        f'{greek_name}<br>'
                        'S/K: %{x:.3f}<br>'
                        'T: %{y:.3f}<br>'
                        'Value: %{z:.4f}<extra></extra>'
                    )
                ),
                row=row, col=col
            )

            # Add ATM line
            atm_idx = np.argmin(np.abs(moneyness - 1.0))
            atm_values = surface[:, atm_idx]
            fig.add_trace(
                go.Scatter3d(
                    x=[1.0] * len(time),
                    y=time,
                    z=atm_values,
                    mode='lines',
                    line=dict(color='lime', width=4),
                    showlegend=(idx == 0),
                    name='ATM'
                ),
                row=row, col=col
            )

            # Add current position
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
                fig.add_trace(
                    go.Scatter3d(
                        x=[current_S / self.K],
                        y=[current_T],
                        z=[value],
                        mode='markers',
                        marker=dict(size=8, color='red'),
                        showlegend=(idx == 0),
                        name='Current'
                    ),
                    row=row, col=col
                )

        # Update all scene layouts
        for i in range(1, 5):
            scene_name = f'scene{i}' if i > 1 else 'scene'
            fig.update_layout(**{
                scene_name: dict(
                    xaxis=dict(title='S/K', backgroundcolor='rgb(20,20,20)', gridcolor='rgb(50,50,50)'),
                    yaxis=dict(title='T (Years)', backgroundcolor='rgb(20,20,20)', gridcolor='rgb(50,50,50)'),
                    zaxis=dict(title='', backgroundcolor='rgb(20,20,20)', gridcolor='rgb(50,50,50)'),
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                    aspectratio=dict(x=1, y=1, z=0.7)
                )
            })

        # Formulas annotation
        formulas = (
            "Black-Scholes PDE: Θ + ½σ²S²Γ + rSΔ = rV<br>"
            f"Parameters: K={self.K}, r={self.r:.2%}, σ={self.sigma:.2%}"
        )

        fig.update_layout(
            title=dict(
                text=f"<b>Black-Scholes Greek Surfaces - {option_type.capitalize()} Option</b><br><sup>{formulas}</sup>",
                x=0.5,
                xanchor='center',
                font=dict(size=18)
            ),
            paper_bgcolor='rgb(10, 10, 10)',
            font=dict(color='white'),
            height=900,
            width=1400,
            legend=dict(
                x=0.02, y=0.02,
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='rgba(255,255,255,0.3)'
            ),
            margin=dict(l=0, r=0, t=100, b=0)
        )

        return fig

    def plot_greeks_comparison(
        self,
        current_S: float = 100,
        current_T: float = 0.5
    ) -> go.Figure:
        """
        Create a comparison plot showing call vs put Greeks.

        Parameters:
        -----------
        current_S : float
            Current spot price
        current_T : float
            Current time to expiration

        Returns:
        --------
        go.Figure : Plotly figure comparing call and put Greeks
        """
        greeks = ['delta', 'gamma', 'theta', 'vega']

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}],
                   [{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=[
                f"{self.GREEK_INFO[g]['name']} (Call vs Put)" for g in greeks
            ],
            horizontal_spacing=0.05,
            vertical_spacing=0.1
        )

        for idx, greek in enumerate(greeks):
            row = idx // 2 + 1
            col = idx % 2 + 1

            # Calculate both call and put surfaces
            moneyness, time, surface_call = self._calculate_surface(greek, 'call')
            _, _, surface_put = self._calculate_surface(greek, 'put')

            M, T = np.meshgrid(moneyness, time)

            # Add call surface
            fig.add_trace(
                go.Surface(
                    x=M, y=T, z=surface_call,
                    colorscale='Blues',
                    showscale=False,
                    opacity=0.7,
                    name=f'{greek} Call'
                ),
                row=row, col=col
            )

            # Add put surface (for Greeks that differ)
            if greek in ['delta', 'theta', 'rho']:
                fig.add_trace(
                    go.Surface(
                        x=M, y=T, z=surface_put,
                        colorscale='Reds',
                        showscale=False,
                        opacity=0.7,
                        name=f'{greek} Put'
                    ),
                    row=row, col=col
                )

        # Update layouts
        for i in range(1, 5):
            scene_name = f'scene{i}' if i > 1 else 'scene'
            fig.update_layout(**{
                scene_name: dict(
                    xaxis=dict(title='S/K'),
                    yaxis=dict(title='T'),
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
                )
            })

        fig.update_layout(
            title=dict(
                text="<b>Call vs Put Greek Surfaces Comparison</b>",
                x=0.5,
                font=dict(size=18)
            ),
            height=900,
            width=1400,
            paper_bgcolor='rgb(10, 10, 10)',
            font=dict(color='white')
        )

        return fig

    def create_animation(
        self,
        greek: str = 'delta',
        option_type: str = 'call',
        parameter: str = 'sigma',
        param_range: Optional[np.ndarray] = None
    ) -> go.Figure:
        """
        Create an animated surface showing how Greek changes with a parameter.

        Parameters:
        -----------
        greek : str
            Greek to animate
        option_type : str
            'call' or 'put'
        parameter : str
            Parameter to vary ('sigma', 'r', or 'K')
        param_range : np.ndarray, optional
            Range of parameter values

        Returns:
        --------
        go.Figure : Animated Plotly figure
        """
        if param_range is None:
            if parameter == 'sigma':
                param_range = np.linspace(0.1, 0.5, 20)
            elif parameter == 'r':
                param_range = np.linspace(0.01, 0.1, 20)
            elif parameter == 'K':
                param_range = np.linspace(80, 120, 20)

        frames = []
        initial_surface = None

        for i, param_val in enumerate(param_range):
            # Update parameter
            if parameter == 'sigma':
                viz = GreekSurfaceVisualizer(self.K, self.r, param_val, self.q)
            elif parameter == 'r':
                viz = GreekSurfaceVisualizer(self.K, param_val, self.sigma, self.q)
            elif parameter == 'K':
                viz = GreekSurfaceVisualizer(param_val, self.r, self.sigma, self.q)

            moneyness, time, surface = viz._calculate_surface(greek, option_type)
            M, T = np.meshgrid(moneyness, time)

            if i == 0:
                initial_surface = surface

            frames.append(go.Frame(
                data=[go.Surface(
                    x=M, y=T, z=surface,
                    colorscale=self.COLORSCALES.get(greek, 'Viridis')
                )],
                name=f'{parameter}={param_val:.3f}'
            ))

        # Create figure with initial frame
        M, T = np.meshgrid(self.moneyness_range, self.time_range)
        fig = go.Figure(
            data=[go.Surface(
                x=M, y=T, z=initial_surface,
                colorscale=self.COLORSCALES.get(greek, 'Viridis')
            )],
            frames=frames
        )

        # Add animation controls
        fig.update_layout(
            title=dict(
                text=f"<b>{self.GREEK_INFO[greek]['name']} vs {parameter}</b>",
                x=0.5
            ),
            scene=dict(
                xaxis_title='S/K',
                yaxis_title='T (Years)',
                zaxis_title=greek.capitalize(),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'y': 0,
                'x': 0.1,
                'xanchor': 'right',
                'yanchor': 'top',
                'buttons': [
                    {
                        'label': '▶ Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 100},
                            'fromcurrent': True
                        }]
                    },
                    {
                        'label': '⏸ Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0},
                            'mode': 'immediate'
                        }]
                    }
                ]
            }],
            sliders=[{
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'prefix': f'{parameter} = ',
                    'visible': True,
                    'xanchor': 'right'
                },
                'steps': [
                    {
                        'label': f'{v:.3f}',
                        'method': 'animate',
                        'args': [[f'{parameter}={v:.3f}'], {
                            'frame': {'duration': 0},
                            'mode': 'immediate'
                        }]
                    }
                    for v in param_range
                ]
            }],
            paper_bgcolor='rgb(10, 10, 10)',
            font=dict(color='white'),
            height=700,
            width=1000
        )

        return fig

    def save_html(self, fig: go.Figure, filename: str) -> None:
        """Save figure as interactive HTML file."""
        fig.write_html(filename, include_plotlyjs=True, full_html=True)

    def save_image(
        self,
        fig: go.Figure,
        filename: str,
        format: str = 'png',
        width: int = 1400,
        height: int = 900,
        scale: int = 2
    ) -> None:
        """Save figure as static image."""
        fig.write_image(filename, format=format, width=width, height=height, scale=scale)


if __name__ == "__main__":
    # Demo
    print("Creating Greek Surface Visualizations...")

    viz = GreekSurfaceVisualizer(K=100, r=0.05, sigma=0.2)

    # Create single Greek surface
    fig = viz.plot_greek_surface(
        greek='delta',
        option_type='call',
        current_S=100,
        current_T=0.5,
        show_atm_line=True,
        show_peak=True
    )

    print("Saving delta surface...")
    viz.save_html(fig, 'delta_surface.html')

    # Create all Greeks
    fig_all = viz.plot_all_greeks(option_type='call', current_S=100, current_T=0.5)
    viz.save_html(fig_all, 'all_greeks_surface.html')

    print("Done! Open the HTML files in your browser.")
