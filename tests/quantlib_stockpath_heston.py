import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt

# 1. Set date and global evaluation date
valuation_date = ql.Date(19, 5, 2025)
ql.Settings.instance().evaluationDate = valuation_date

# 2. Market parameters
spot_price = 100.0
risk_free_rate = 0.05
dividend_yield = 0.02
v0 = 0.04
kappa = 2.0
theta = 0.04
sigma = 0.3
rho = -0.7

# 3. Yield and dividend curves
day_count = ql.Actual365Fixed()
risk_free_curve = ql.FlatForward(valuation_date, risk_free_rate, day_count)
dividend_curve = ql.FlatForward(valuation_date, dividend_yield, day_count)
flat_risk_free_handle = ql.YieldTermStructureHandle(risk_free_curve)
flat_dividend_handle = ql.YieldTermStructureHandle(dividend_curve)

# 4. Heston model
spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
heston_process = ql.HestonProcess(
    flat_risk_free_handle,
    flat_dividend_handle,
    spot_handle,
    v0, kappa, theta, sigma, rho
)
heston_model = ql.HestonModel(heston_process)

# 5. Monte Carlo setup
time_steps = 252
num_paths = 1000
horizon = 1.0
seed = 42

# Fix: Use Sobol sequence with correct dimension
dimension = time_steps * 2  # Two factors for Heston
rng = ql.GaussianRandomSequenceGenerator(
    ql.SobolRsg(dimension, seed)
)
path_generator = ql.GaussianPathGenerator(
    heston_process, horizon, time_steps, rng, False
)

# 6. Generate paths
paths = np.zeros((num_paths, time_steps + 1))
for i in range(num_paths):
    path = path_generator.next().value()
    paths[i, :] = [path[j] for j in range(time_steps + 1)]

# 7. Visualize
time_grid = np.linspace(0, horizon, time_steps + 1)
plt.figure(figsize=(10, 6))
for i in range(min(num_paths, 50)):
    plt.plot(time_grid, paths[i, :], lw=0.5, alpha=0.5)
plt.title('Heston Model Stock Price Paths')
plt.xlabel('Time (Years)')
plt.ylabel('Stock Price')
plt.grid(True)
plt.show()