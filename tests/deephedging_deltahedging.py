#%%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time
import os
from scipy.stats import norm
import scipy.optimize as optimize

#%%
# Set random seed to ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)
# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#%%
# Heston model parameters
class HestonParams:
    def __init__(self):
        self.S0 = 100.0       # Initial stock price
        self.V0 = 0.04        # Initial volatility
        self.kappa = 2.0      # Mean reversion speed of volatility
        self.theta = 0.04     # Long-term mean of volatility
        self.sigma = 0.3      # Volatility of volatility
        self.rho = -0.7       # Correlation between price and volatility
        self.r = 0.05         # Risk-free rate
        self.T = 1.0          # Option maturity (years)
        self.K = 100.0        # Option strike price
        self.trading_days = 252  # Trading days per year

# %%
# Simulate Heston model
def simulate_heston(params, n_paths, n_steps):
    dt = params.T / n_steps
    sqrt_dt = np.sqrt(dt)

    # Initialize paths
    S = np.zeros((n_paths, n_steps + 1))
    V = np.zeros((n_paths, n_steps + 1))
    integrated_var = np.zeros((n_paths, n_steps + 1))

    # Set initial values
    S[:, 0] = params.S0
    V[:, 0] = params.V0

    # Generate correlated random numbers using PyTorch
    Z1 = torch.randn(n_paths, n_steps).numpy()
    Z2 = params.rho * Z1 + np.sqrt(1 - params.rho ** 2) * torch.randn(n_paths, n_steps).numpy()

    # Simulate paths
    for t in range(n_steps):
        # Ensure volatility is positive
        V[:, t] = np.maximum(V[:, t], 0)

        # Update stock price
        S[:, t + 1] = S[:, t] * np.exp((params.r - 0.5 * V[:, t]) * dt + np.sqrt(V[:, t]) * sqrt_dt * Z1[:, t])

        # Update volatility
        V[:, t + 1] = V[:, t] + params.kappa * (params.theta - V[:, t]) * dt + params.sigma * np.sqrt(
            V[:, t]) * sqrt_dt * Z2[:, t]

        # Calculate integrated variance
        integrated_var[:, t + 1] = integrated_var[:, t] + V[:, t] * dt

    return S, V, integrated_var

#%%
# Calculate European put option payoff
def put_option_payoff(S, K):
    if isinstance(S, torch.Tensor):
        return torch.maximum(torch.tensor(K, device=S.device) - S, torch.tensor(0.0, device=S.device))
    else:
        return np.maximum(K - S, 0)


# %%
# Heston option pricing and hedging functions

# Heston model characteristic function
def heston_characteristic_function(phi, S, V, T, params):
    a = params.kappa * params.theta
    b = params.kappa

    d = np.sqrt((params.rho * params.sigma * phi * 1j - b) ** 2 + (params.sigma ** 2) * (phi * 1j + phi ** 2))
    g = (b - params.rho * params.sigma * phi * 1j - d) / (b - params.rho * params.sigma * phi * 1j + d)

    exp1 = np.exp(params.r * T * phi * 1j)

    exp2 = np.exp(a * params.T * (b - params.rho * params.sigma * phi * 1j - d) / (params.sigma ** 2))

    exp3 = np.exp((V * (b - params.rho * params.sigma * phi * 1j - d)) / (params.sigma ** 2 * (1 - g * np.exp(-d * T))))

    return exp1 * exp2 * exp3

# %%
# Calculate option price under Heston model using numerical methods
def heston_option_price_fft(S, V, K, T, params, option_type='put'):
    import numpy as np
    from scipy.integrate import quad
    from scipy import interpolate

    # Characteristic function integral
    def integrand_call(phi, S, V, K, T, params):
        numerator = np.exp(-phi * np.log(K) * 1j) * heston_characteristic_function(phi - 1j, S, V, T, params)
        denominator = phi * 1j
        return np.real(numerator / denominator)

    # Calculate integral
    result, _ = quad(integrand_call, 0, 100, args=(S, V, K, T, params), limit=100)
    call_price = S / 2 + result / np.pi

    # Use put-call parity to get put option price
    if option_type.lower() == 'call':
        return call_price
    else:  # Put option
        return call_price - S + K * np.exp(-params.r * T)

# %%
# Calculate option Delta under Heston model
def heston_option_delta(S, V, K, T, params, option_type='put'):
    h = 0.01  # Small price change for numerical differentiation

    price_up = heston_option_price_fft(S + h, V, K, T, params, option_type)
    price_down = heston_option_price_fft(S - h, V, K, T, params, option_type)

    delta = (price_up - price_down) / (2 * h)
    return delta


# %%
# Calculate option Vega (sensitivity to variance) under Heston model
def heston_option_vega(S, V, K, T, params, option_type='put'):
    h = 0.0001  # Small variance change for numerical differentiation

    price_up = heston_option_price_fft(S, V + h, K, T, params, option_type)
    price_down = heston_option_price_fft(S, V - h, K, T, params, option_type)

    vega = (price_up - price_down) / (2 * h)
    return vega


# %%
# Calculate variance swap value based on formula (31)
def calculate_variance_swap(integrated_var_t, V_t, t, T, params):
    """
    Calculate variance swap value based on S_t^2 = ∫_0^t V_s ds + L(t, V_t)
    where L(t,v) = (v-h)/α * (1-e^(-α(T-t))) + h(T-t)

    Parameters:
    integrated_var_t - Realized variance from 0 to t
    V_t - Current instantaneous variance
    t - Current time
    T - Maturity
    params - Model parameters

    Returns:
    variance_swap_value - Variance swap value
    """
    # Calculate L(t, V_t) term
    h = params.theta  # Long-term variance mean
    alpha = params.kappa  # Mean reversion rate

    remaining_T = T - t
    L_t_v = (V_t - h) / alpha * (1 - np.exp(-alpha * remaining_T)) + h * remaining_T

    # Variance swap value = realized variance + expected future variance
    variance_swap_value = integrated_var_t + L_t_v

    return variance_swap_value

# %%
# Calculate sensitivity of variance swap to instantaneous variance
def variance_swap_sensitivity(V_t, t, T, params):
    """
    Calculate ∂L(t,V_t)/∂V_t for hedging purposes

    Parameters:
    V_t - Current instantaneous variance
    t - Current time
    T - Maturity
    params - Model parameters

    Returns:
    sensitivity - Partial derivative of L with respect to V
    """
    alpha = params.kappa
    remaining_T = T - t

    # From the formula: ∂L(t,v)/∂v = 1/α * (1-e^(-α(T-t)))
    sensitivity = (1 - np.exp(-alpha * remaining_T)) / alpha

    return sensitivity

# %%
# Evaluate Delta-Vega hedging strategy based on Formulas (32) and (33)
def evaluate_delta_vega_hedging(params, n_paths=1000, n_steps=30, use_vega=True):
    """
    Evaluate Delta-Vega hedging strategy under Heston model using formula (33)
    δ¹ := ∂u(t,S¹,Vt)/∂S¹ and δ² := ∂u(t,S¹,Vt)/∂L(t,Vt)

    Parameters:
    params: Heston model parameters
    n_paths: Number of simulation paths
    n_steps: Number of hedging steps
    use_vega: Whether to use Vega hedging (if False, only Delta hedging)

    Returns:
    final_portfolio: Final portfolio values
    delta_positions: Delta hedging positions
    vega_positions: Vega hedging positions
    """
    strategy_name = "Delta-Vega Hedging" if use_vega else "Delta-Only Hedging"
    print(f"\nStarting {strategy_name} evaluation...")

    # Generate price paths
    print("Generating price paths...")
    S, V, integrated_var = simulate_heston(params, n_paths, n_steps)
    dt = params.T / n_steps

    # Initialize portfolio and positions
    portfolio_values = np.zeros(n_paths)
    delta_positions = np.zeros((n_paths, n_steps + 1))
    vega_positions = np.zeros((n_paths, n_steps + 1))
    trading_costs = np.zeros(n_paths)

    # Calculate option payoffs
    option_payoff_np = put_option_payoff(S[:, -1], params.K)

    # Set up progress monitoring
    print(f"Starting hedging calculations for {n_paths} paths over {n_steps} steps...")
    total_calculations = n_steps * n_paths
    progress_interval = max(1, n_paths // 10)
    last_progress_time = time.time()
    start_time = time.time()

    # Hedge at each time step
    for t in range(n_steps):
        curr_t = t * params.T / n_steps
        remaining_T = params.T - curr_t

        print(f"Time step { t +1}/{n_steps} - Remaining T: {remaining_T:.4f}")
        paths_done = 0

        # Process each path
        for i in range(n_paths):
            # Calculate Delta (Formula 33: δ¹ := ∂u(t,S¹,Vt)/∂S¹)
            delta = heston_option_delta(S[i, t], V[i, t], params.K, remaining_T, params, 'put')
            delta_trade = -delta - delta_positions[i, t]  # New position minus old position
            delta_positions[i, t+ 1] = -delta  # Negative delta for hedge

            # If using Vega hedging
            if use_vega:
                # Calculate option's sensitivity to variance (Formula 33)
                option_dv_sensitivity = heston_option_vega(S[i, t], V[i, t], params.K, remaining_T, params, 'put')

                # Calculate sensitivity of variance swap to instantaneous variance (∂L(t,Vt)/∂Vt)
                vs_sensitivity = variance_swap_sensitivity(V[i, t], curr_t, params.T, params)

                # Calculate vega hedge ratio according to formula (33): δ² := ∂u(t,S¹,Vt)/∂L(t,Vt)
                # Here we convert from ∂u/∂V to ∂u/∂L using the chain rule: (∂u/∂V) = (∂u/∂L) * (∂L/∂V)
                # So δ² = (∂u/∂V) / (∂L/∂V)
                vega_hedge_ratio = -option_dv_sensitivity / vs_sensitivity if vs_sensitivity != 0 else 0

                vega_trade = vega_hedge_ratio - vega_positions[i, t]
                vega_positions[i, t + 1] = vega_hedge_ratio
            else:
                vega_trade = 0
                vega_positions[i, t + 1] = 0

            # Calculate trading costs
            cost = 0.001 * (np.abs(delta_trade) * S[i, t] + np.abs(vega_trade))
            trading_costs[i] += cost

            # Calculate variance swap values for current and next time step
            var_swap_t = calculate_variance_swap(
                integrated_var[i, t], V[i, t], curr_t, params.T, params
            )

            var_swap_t_plus_dt = calculate_variance_swap(
                integrated_var[i, t + 1], V[i, t + 1], curr_t + dt, params.T, params
            )

            # Calculate PnL
            delta_pnl = delta_positions[i, t] * (S[i, t + 1] - S[i, t])
            vega_pnl = vega_positions[i, t] * (var_swap_t_plus_dt - var_swap_t)

            # Update portfolio value
            portfolio_values[i] += delta_pnl + vega_pnl - cost

            paths_done += 1

            # Update progress
            if paths_done % progress_interval == 0 or paths_done == n_paths:
                current_time = time.time()
                elapsed = current_time - start_time
                progress_pct = (t * n_paths + paths_done) / total_calculations * 100

                # Only update if at least 1 second has passed since last update
                if current_time - last_progress_time >= 1.0:
                    last_progress_time = current_time

                    # Estimate time remaining
                    if progress_pct > 0:
                        total_estimated_time = elapsed / (progress_pct / 100)
                        remaining_time = total_estimated_time - elapsed

                        print(
                            f"  Progress: {progress_pct:.1f}% - Paths: {paths_done}/{n_paths} in step {t + 1}/{n_steps}")
                        print(f"  Elapsed time: {elapsed:.1f}s, Est. remaining: {remaining_time:.1f}s")

    # Final portfolio value (minus option payoff)
    final_portfolio = portfolio_values - option_payoff_np

    # Calculate statistics
    mean_pnl = np.mean(final_portfolio)
    std_pnl = np.std(final_portfolio)

    # Calculate VaR and CVaR
    alpha = 0.05
    sorted_pnl = np.sort(final_portfolio)
    var_index = int(np.ceil(alpha * n_paths)) - 1
    var_index = max(0, var_index)
    var = max(0, -sorted_pnl[var_index])
    cvar = max(0, -np.mean(sorted_pnl[:var_index + 1]))

    total_time = time.time() - start_time
    print(f"\n{strategy_name} calculation complete in {total_time:.2f} seconds")
    print(f"{strategy_name} Evaluation Results:")
    print(f"Mean P&L: {mean_pnl:.4f}")
    print(f"P&L Standard Deviation: {std_pnl:.4f}")
    print(f"VaR(95%): {var:.4f}")
    print(f"CVaR(95%): {cvar:.4f}")
    print(f"Average Trading Cost: {np.mean(trading_costs):.4f}")

    return final_portfolio, delta_positions, vega_positions

if __name__ == '__main__':
    params = HestonParams()
    n_steps = 30  # Number of hedging steps (30 trading days)
    n_paths = 500
    # Heston Delta-only hedging
    print("\nEvaluating Heston Delta-only hedging strategy:")
    d_pnl, d_delta, _ = evaluate_delta_vega_hedging(params, n_paths, n_steps, use_vega=False)
    # Heston Delta-Vega hedging
    print("\nEvaluating Heston Delta-Vega hedging strategy:")
    dv_pnl, dv_delta, dv_vega = evaluate_delta_vega_hedging(params, n_paths, n_steps, use_vega=True)
