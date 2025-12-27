"""
Black-Scholes Option Pricing Model with Complete Greeks Calculations

This module implements the Black-Scholes-Merton model for European option pricing
and calculates all first and second-order Greeks (sensitivities).

Mathematical Foundation:
-----------------------
The Black-Scholes PDE:
    ∂V/∂t + ½σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0

Which can be rewritten in terms of Greeks as:
    Θ + ½σ²S²Γ + rSΔ = rV

Where:
    V = Option value
    S = Underlying asset price
    K = Strike price
    r = Risk-free interest rate
    σ = Volatility
    τ = Time to expiration (T - t)
    Φ = Standard normal CDF
    φ = Standard normal PDF
"""

import numpy as np
from scipy.stats import norm
from typing import Union, Tuple, Literal
from dataclasses import dataclass


@dataclass
class GreeksResult:
    """Container for all Greek values."""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    # Second-order Greeks
    vanna: float  # ∂Δ/∂σ = ∂V/∂σ∂S
    volga: float  # ∂²V/∂σ² (also called Vomma)
    charm: float  # ∂Δ/∂τ (Delta decay)
    veta: float   # ∂V/∂σ∂τ (Vega decay)
    speed: float  # ∂Γ/∂S
    zomma: float  # ∂Γ/∂σ
    color: float  # ∂Γ/∂τ


class BlackScholes:
    """
    Black-Scholes-Merton Option Pricing Model

    Implements European option pricing and all Greeks calculations.
    Supports both call and put options.

    Parameters:
    -----------
    S : float
        Current price of the underlying asset
    K : float
        Strike price of the option
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility of the underlying asset (annualized)
    q : float, optional
        Continuous dividend yield (default: 0)

    Attributes:
    -----------
    d1, d2 : float
        Intermediate values used in BS formula

    Example:
    --------
    >>> bs = BlackScholes(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
    >>> print(f"Call Price: ${bs.call_price():.2f}")
    >>> print(f"Delta: {bs.delta('call'):.4f}")
    """

    # Small epsilon to prevent division by zero
    EPSILON_TIME = 1e-10
    EPSILON_SIGMA = 1e-10

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ):
        self.S = S
        self.K = K
        self.T = max(T, self.EPSILON_TIME)  # Prevent division by zero
        self.r = r
        self.sigma = max(sigma, self.EPSILON_SIGMA)
        self.q = q

        # Calculate d1 and d2
        self._calculate_d1_d2()

    def _calculate_d1_d2(self) -> None:
        """
        Calculate d1 and d2 parameters.

        d₁ = [ln(S/K) + (r - q + σ²/2)τ] / (σ√τ)
        d₂ = d₁ - σ√τ
        """
        sqrt_T = np.sqrt(self.T)
        self.d1 = (
            np.log(self.S / self.K) +
            (self.r - self.q + 0.5 * self.sigma**2) * self.T
        ) / (self.sigma * sqrt_T)
        self.d2 = self.d1 - self.sigma * sqrt_T

    @staticmethod
    def _N(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Standard normal cumulative distribution function Φ(x)."""
        return norm.cdf(x)

    @staticmethod
    def _n(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Standard normal probability density function φ(x)."""
        return norm.pdf(x)

    # ==================== OPTION PRICES ====================

    def call_price(self) -> float:
        """
        Calculate European call option price.

        C = S·e^(-qτ)·Φ(d₁) - K·e^(-rτ)·Φ(d₂)

        Returns:
        --------
        float : Call option price
        """
        return (
            self.S * np.exp(-self.q * self.T) * self._N(self.d1) -
            self.K * np.exp(-self.r * self.T) * self._N(self.d2)
        )

    def put_price(self) -> float:
        """
        Calculate European put option price.

        P = K·e^(-rτ)·Φ(-d₂) - S·e^(-qτ)·Φ(-d₁)

        Returns:
        --------
        float : Put option price
        """
        return (
            self.K * np.exp(-self.r * self.T) * self._N(-self.d2) -
            self.S * np.exp(-self.q * self.T) * self._N(-self.d1)
        )

    def price(self, option_type: Literal['call', 'put'] = 'call') -> float:
        """Get option price based on type."""
        if option_type.lower() == 'call':
            return self.call_price()
        return self.put_price()

    # ==================== FIRST-ORDER GREEKS ====================

    def delta(self, option_type: Literal['call', 'put'] = 'call') -> float:
        """
        Calculate Delta (Δ) - sensitivity to underlying price.

        Δ_call = e^(-qτ)·Φ(d₁)
        Δ_put = e^(-qτ)·[Φ(d₁) - 1] = -e^(-qτ)·Φ(-d₁)

        Delta represents the rate of change of option value with respect
        to changes in the underlying asset's price. It's also interpreted
        as the probability of the option expiring in-the-money (ITM).

        Parameters:
        -----------
        option_type : str
            'call' or 'put'

        Returns:
        --------
        float : Delta value [-1, 1]
        """
        discount = np.exp(-self.q * self.T)
        if option_type.lower() == 'call':
            return discount * self._N(self.d1)
        return discount * (self._N(self.d1) - 1)

    def gamma(self) -> float:
        """
        Calculate Gamma (Γ) - rate of change of Delta.

        Γ = e^(-qτ)·φ(d₁) / (S·σ·√τ)

        Gamma is the same for both calls and puts. It measures the
        convexity of the option value with respect to the underlying price.
        High gamma means delta is very sensitive to price changes.

        Returns:
        --------
        float : Gamma value (always positive)
        """
        return (
            np.exp(-self.q * self.T) * self._n(self.d1) /
            (self.S * self.sigma * np.sqrt(self.T))
        )

    def theta(self, option_type: Literal['call', 'put'] = 'call') -> float:
        """
        Calculate Theta (Θ) - time decay.

        For a call:
        Θ_call = -[S·σ·e^(-qτ)·φ(d₁)] / (2√τ)
                 - r·K·e^(-rτ)·Φ(d₂) + q·S·e^(-qτ)·Φ(d₁)

        For a put:
        Θ_put = -[S·σ·e^(-qτ)·φ(d₁)] / (2√τ)
                + r·K·e^(-rτ)·Φ(-d₂) - q·S·e^(-qτ)·Φ(-d₁)

        Theta represents the rate of decline in option value due to
        passage of time (all else equal). Usually negative for long options.

        Parameters:
        -----------
        option_type : str
            'call' or 'put'

        Returns:
        --------
        float : Theta value (typically negative, per year)
        """
        sqrt_T = np.sqrt(self.T)
        discount_q = np.exp(-self.q * self.T)
        discount_r = np.exp(-self.r * self.T)

        # Common term (time decay from volatility)
        common = -(self.S * self.sigma * discount_q * self._n(self.d1)) / (2 * sqrt_T)

        if option_type.lower() == 'call':
            return (
                common -
                self.r * self.K * discount_r * self._N(self.d2) +
                self.q * self.S * discount_q * self._N(self.d1)
            )
        return (
            common +
            self.r * self.K * discount_r * self._N(-self.d2) -
            self.q * self.S * discount_q * self._N(-self.d1)
        )

    def theta_per_day(self, option_type: Literal['call', 'put'] = 'call') -> float:
        """Return theta divided by 365 (daily decay)."""
        return self.theta(option_type) / 365

    def vega(self) -> float:
        """
        Calculate Vega (V) - sensitivity to volatility.

        V = S·e^(-qτ)·√τ·φ(d₁)

        Vega is the same for both calls and puts. It measures how much
        the option price changes for a 1% change in implied volatility.

        Note: Vega is often quoted per 1% change in volatility (divide by 100).

        Returns:
        --------
        float : Vega value (always positive)
        """
        return (
            self.S * np.exp(-self.q * self.T) *
            np.sqrt(self.T) * self._n(self.d1)
        )

    def vega_pct(self) -> float:
        """Return vega per 1% change in volatility."""
        return self.vega() / 100

    def rho(self, option_type: Literal['call', 'put'] = 'call') -> float:
        """
        Calculate Rho (ρ) - sensitivity to interest rate.

        ρ_call = K·τ·e^(-rτ)·Φ(d₂)
        ρ_put = -K·τ·e^(-rτ)·Φ(-d₂)

        Rho measures the sensitivity of option value to changes in the
        risk-free interest rate.

        Note: Rho is often quoted per 1% change in rate (divide by 100).

        Parameters:
        -----------
        option_type : str
            'call' or 'put'

        Returns:
        --------
        float : Rho value
        """
        discount = self.K * self.T * np.exp(-self.r * self.T)
        if option_type.lower() == 'call':
            return discount * self._N(self.d2)
        return -discount * self._N(-self.d2)

    def rho_pct(self, option_type: Literal['call', 'put'] = 'call') -> float:
        """Return rho per 1% change in interest rate."""
        return self.rho(option_type) / 100

    # ==================== SECOND-ORDER GREEKS ====================

    def vanna(self) -> float:
        """
        Calculate Vanna - sensitivity of delta to volatility.

        Vanna = ∂Δ/∂σ = ∂V/∂σ∂S = -e^(-qτ)·φ(d₁)·d₂/(σ)

        Also known as DdeltaDvol. Measures how delta changes when
        volatility changes, or equivalently, how vega changes when
        spot price changes.

        Returns:
        --------
        float : Vanna value
        """
        return (
            -np.exp(-self.q * self.T) * self._n(self.d1) *
            self.d2 / self.sigma
        )

    def volga(self) -> float:
        """
        Calculate Volga (Vomma) - sensitivity of vega to volatility.

        Volga = ∂²V/∂σ² = Vega · d₁·d₂/σ

        Measures the convexity of the option value with respect to
        volatility. Important for understanding volatility smile dynamics.

        Returns:
        --------
        float : Volga value
        """
        return self.vega() * self.d1 * self.d2 / self.sigma

    def charm(self, option_type: Literal['call', 'put'] = 'call') -> float:
        """
        Calculate Charm (Delta Bleed) - rate of change of delta over time.

        Charm = ∂Δ/∂τ = -∂Δ/∂t

        For calls:
        Charm = q·e^(-qτ)·Φ(d₁) - e^(-qτ)·φ(d₁)·[2(r-q)τ - d₂σ√τ]/(2τσ√τ)

        Measures how delta changes as time passes. Important for
        understanding how hedges need to be adjusted over time.

        Returns:
        --------
        float : Charm value (per year)
        """
        sqrt_T = np.sqrt(self.T)
        discount_q = np.exp(-self.q * self.T)
        n_d1 = self._n(self.d1)

        if option_type.lower() == 'call':
            return (
                self.q * discount_q * self._N(self.d1) -
                discount_q * n_d1 *
                (2 * (self.r - self.q) * self.T - self.d2 * self.sigma * sqrt_T) /
                (2 * self.T * self.sigma * sqrt_T)
            )
        return (
            -self.q * discount_q * self._N(-self.d1) -
            discount_q * n_d1 *
            (2 * (self.r - self.q) * self.T - self.d2 * self.sigma * sqrt_T) /
            (2 * self.T * self.sigma * sqrt_T)
        )

    def veta(self) -> float:
        """
        Calculate Veta - rate of change of vega over time.

        Veta = ∂V/∂τ = -∂Vega/∂t

        Measures how vega changes as time passes. Important for
        understanding how volatility exposure changes over time.

        Returns:
        --------
        float : Veta value (per year)
        """
        sqrt_T = np.sqrt(self.T)
        discount_q = np.exp(-self.q * self.T)

        term1 = self.q
        term2 = (
            (self.r - self.q) * self.d1 / (self.sigma * sqrt_T) -
            (1 + self.d1 * self.d2) / (2 * self.T)
        )

        return (
            -self.S * discount_q * self._n(self.d1) * sqrt_T *
            (term1 + term2)
        )

    def speed(self) -> float:
        """
        Calculate Speed - rate of change of gamma with respect to spot.

        Speed = ∂Γ/∂S = -Γ/S · (1 + d₁/(σ√τ))

        Third derivative of option price with respect to spot.
        Measures how gamma changes when the underlying moves.

        Returns:
        --------
        float : Speed value
        """
        sqrt_T = np.sqrt(self.T)
        return -self.gamma() / self.S * (1 + self.d1 / (self.sigma * sqrt_T))

    def zomma(self) -> float:
        """
        Calculate Zomma - rate of change of gamma with respect to volatility.

        Zomma = ∂Γ/∂σ = Γ · (d₁·d₂ - 1)/σ

        Measures how gamma changes when volatility changes.

        Returns:
        --------
        float : Zomma value
        """
        return self.gamma() * (self.d1 * self.d2 - 1) / self.sigma

    def color(self) -> float:
        """
        Calculate Color (Gamma Bleed) - rate of change of gamma over time.

        Color = ∂Γ/∂τ = -∂Γ/∂t

        Measures how gamma changes as time passes.

        Returns:
        --------
        float : Color value (per year)
        """
        sqrt_T = np.sqrt(self.T)
        discount_q = np.exp(-self.q * self.T)
        n_d1 = self._n(self.d1)

        term = (
            2 * (self.r - self.q) * self.T - self.d2 * self.sigma * sqrt_T
        ) / (2 * self.T * self.sigma * sqrt_T)

        return (
            -discount_q * n_d1 / (2 * self.S * self.T * self.sigma * sqrt_T) *
            (2 * self.q * self.T + 1 + self.d1 * term)
        )

    # ==================== UTILITY METHODS ====================

    def all_greeks(self, option_type: Literal['call', 'put'] = 'call') -> GreeksResult:
        """
        Calculate all Greeks at once.

        Returns:
        --------
        GreeksResult : Dataclass containing all Greek values
        """
        return GreeksResult(
            delta=self.delta(option_type),
            gamma=self.gamma(),
            theta=self.theta(option_type),
            vega=self.vega(),
            rho=self.rho(option_type),
            vanna=self.vanna(),
            volga=self.volga(),
            charm=self.charm(option_type),
            veta=self.veta(),
            speed=self.speed(),
            zomma=self.zomma(),
            color=self.color()
        )

    def implied_volatility(
        self,
        market_price: float,
        option_type: Literal['call', 'put'] = 'call',
        tol: float = 1e-6,
        max_iter: int = 100
    ) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.

        Parameters:
        -----------
        market_price : float
            Observed market price of the option
        option_type : str
            'call' or 'put'
        tol : float
            Convergence tolerance
        max_iter : int
            Maximum iterations

        Returns:
        --------
        float : Implied volatility
        """
        sigma = 0.2  # Initial guess

        for _ in range(max_iter):
            bs = BlackScholes(self.S, self.K, self.T, self.r, sigma, self.q)
            price = bs.price(option_type)
            vega = bs.vega()

            if abs(price - market_price) < tol:
                return sigma

            if vega < 1e-10:
                break

            sigma = sigma - (price - market_price) / vega
            sigma = max(0.001, min(sigma, 5.0))  # Bound sigma

        return sigma

    @staticmethod
    def calculate_surface(
        S_range: np.ndarray,
        T_range: np.ndarray,
        K: float,
        r: float,
        sigma: float,
        greek: str = 'delta',
        option_type: str = 'call',
        q: float = 0.0
    ) -> np.ndarray:
        """
        Calculate a Greek surface over ranges of S and T.

        Parameters:
        -----------
        S_range : np.ndarray
            Array of spot prices (or S/K ratios if using moneyness)
        T_range : np.ndarray
            Array of times to expiration
        K : float
            Strike price
        r : float
            Risk-free rate
        sigma : float
            Volatility
        greek : str
            Name of the Greek to calculate
        option_type : str
            'call' or 'put'
        q : float
            Dividend yield

        Returns:
        --------
        np.ndarray : 2D array of Greek values (shape: len(T_range) x len(S_range))
        """
        surface = np.zeros((len(T_range), len(S_range)))

        greek_methods = {
            'delta': lambda bs: bs.delta(option_type),
            'gamma': lambda bs: bs.gamma(),
            'theta': lambda bs: bs.theta(option_type),
            'vega': lambda bs: bs.vega(),
            'rho': lambda bs: bs.rho(option_type),
            'vanna': lambda bs: bs.vanna(),
            'volga': lambda bs: bs.volga(),
            'charm': lambda bs: bs.charm(option_type),
            'price': lambda bs: bs.price(option_type)
        }

        if greek.lower() not in greek_methods:
            raise ValueError(f"Unknown Greek: {greek}")

        method = greek_methods[greek.lower()]

        for i, T in enumerate(T_range):
            for j, S in enumerate(S_range):
                bs = BlackScholes(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
                surface[i, j] = method(bs)

        return surface

    def __repr__(self) -> str:
        return (
            f"BlackScholes(S={self.S}, K={self.K}, T={self.T:.4f}, "
            f"r={self.r:.4f}, σ={self.sigma:.4f}, q={self.q:.4f})"
        )


def verify_bs_pde(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0
) -> Tuple[float, float]:
    """
    Verify the Black-Scholes PDE holds: Θ + ½σ²S²Γ + rSΔ = rV

    Parameters:
    -----------
    S, K, T, r, sigma, q : Black-Scholes parameters

    Returns:
    --------
    Tuple[float, float] : (LHS of PDE, RHS of PDE) - should be approximately equal
    """
    bs = BlackScholes(S, K, T, r, sigma, q)

    theta = bs.theta('call')
    gamma = bs.gamma()
    delta = bs.delta('call')
    V = bs.call_price()

    # Note: theta from our function is ∂V/∂T = -∂V/∂t
    # The BS PDE uses ∂V/∂t, so we negate theta
    lhs = -theta + 0.5 * sigma**2 * S**2 * gamma + (r - q) * S * delta - q * V
    rhs = r * V

    return lhs, rhs


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Black-Scholes Option Pricing Model Demo")
    print("=" * 60)

    # Create a Black-Scholes instance
    bs = BlackScholes(S=100, K=100, T=1.0, r=0.05, sigma=0.2)

    print(f"\nParameters: {bs}")
    print(f"\nd1 = {bs.d1:.6f}")
    print(f"d2 = {bs.d2:.6f}")

    print("\n" + "-" * 40)
    print("Option Prices")
    print("-" * 40)
    print(f"Call Price: ${bs.call_price():.4f}")
    print(f"Put Price:  ${bs.put_price():.4f}")

    print("\n" + "-" * 40)
    print("Call Option Greeks")
    print("-" * 40)
    greeks = bs.all_greeks('call')
    print(f"Delta (Δ):  {greeks.delta:.6f}")
    print(f"Gamma (Γ):  {greeks.gamma:.6f}")
    print(f"Theta (Θ):  {greeks.theta:.6f} (per year)")
    print(f"Theta:      {bs.theta_per_day('call'):.6f} (per day)")
    print(f"Vega (V):   {greeks.vega:.6f}")
    print(f"Rho (ρ):    {greeks.rho:.6f}")

    print("\n" + "-" * 40)
    print("Second-Order Greeks")
    print("-" * 40)
    print(f"Vanna:      {greeks.vanna:.6f}")
    print(f"Volga:      {greeks.volga:.6f}")
    print(f"Charm:      {greeks.charm:.6f}")
    print(f"Speed:      {greeks.speed:.6f}")
    print(f"Zomma:      {greeks.zomma:.6f}")
    print(f"Color:      {greeks.color:.6f}")

    print("\n" + "-" * 40)
    print("Verify BS PDE: Θ + ½σ²S²Γ + rSΔ = rV")
    print("-" * 40)
    lhs, rhs = verify_bs_pde(100, 100, 1.0, 0.05, 0.2)
    print(f"LHS: {lhs:.6f}")
    print(f"RHS: {rhs:.6f}")
    print(f"Difference: {abs(lhs - rhs):.2e}")
