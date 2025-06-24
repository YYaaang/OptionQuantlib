import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dask.config import paths
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
n_paths = 20
S0 = 100.0  # Initial stock price
K = 100.0  # Option strike price
V0 = 0.04  # Initial volatility
kappa = 2.0  # Mean reversion speed of volatility
theta = 0.04  # Long-term mean of volatility
sigma = 0.3  # Volatility of volatility
rho = -0.7  # Correlation between price and volatility
r = 0.05  # Risk-free rate
T = 1.0  # Option maturity (years)
trading_days = 30  # Trading days per year
#%%
# Heston model parameters
class HestonParams:
    def __init__(self, S0, V0, kappa, theta, sigma, rho, r, T, K, trading_days):
        self.S0 = S0       # Initial stock price
        self.V0 = V0        # Initial volatility
        self.kappa = kappa      # Mean reversion speed of volatility
        self.theta = theta     # Long-term mean of volatility
        self.sigma = sigma      # Volatility of volatility
        self.rho = rho       # Correlation between price and volatility
        self.r = r         # Risk-free rate
        self.T = T          # Option maturity (years)
        self.K = K        # Option strike price
        self.trading_days = trading_days  # Trading days per year
params = HestonParams(S0, V0, kappa, theta, sigma, rho, r, T, K, trading_days)
#%%
# numpy
# Simulate Heston model
import torch

def simulate_heston(n_paths, n_steps, S0, V0, kappa, theta, rho, sigma, r):
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    # Initialize paths
    S = np.zeros((n_paths, n_steps + 1))
    V = np.zeros((n_paths, n_steps + 1))
    integrated_var = np.zeros((n_paths, n_steps + 1))

    # Set initial values
    S[:, 0] = S0
    V[:, 0] = V0

    # Generate correlated random numbers using PyTorch
    Z1 = torch.randn(n_paths, n_steps).numpy()
    Z2 = rho * Z1 + np.sqrt(1 - rho ** 2) * torch.randn(n_paths, n_steps).numpy()

    # Simulate paths
    for t in range(n_steps):
        # Ensure volatility is positive
        V[:, t] = np.maximum(V[:, t], 0)

        # Update stock price
        S[:, t + 1] = S[:, t] * np.exp((r - 0.5 * V[:, t]) * dt + np.sqrt(V[:, t]) * sqrt_dt * Z1[:, t])

        # Update volatility
        V[:, t + 1] = V[:, t] + kappa * (theta - V[:, t]) * dt + sigma * np.sqrt(V[:, t]) * sqrt_dt * Z2[:, t]

        # Calculate integrated variance
        integrated_var[:, t + 1] = integrated_var[:, t] + V[:, t] * dt

    return S, V, integrated_var


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


# Calculate option Delta under Heston model
def heston_option_delta(S, V, K, T, params, option_type='put'):
    h = 0.00001  # Small price change for numerical differentiation

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
put = heston_option_price_fft(S0, V0, K, T, params, option_type='put')
call = heston_option_price_fft(S0, V0, K, T, params, option_type='call')
delta = heston_option_delta(S0, V0, K, T, params, option_type='put')
print()