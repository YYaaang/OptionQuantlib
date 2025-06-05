#%%
import QuantLib as ql
from src.QlCalendar import QlCalendar
from src.QlStocks import QlStocks

#%%
n_paths = 20000
S0 = 100.0  # Initial stock price
V0 = 0.04  # Initial volatility
kappa = 2.0  # Mean reversion speed of volatility
theta = 0.04  # Long-term mean of volatility
sigma = 0.3  # Volatility of volatility
rho = -0.7  # Correlation between price and volatility
r = 0.05  # Risk-free rate
T = 1.0  # Option maturity (years)
K = 100.0  # Option strike price
trading_days = 252  # Trading days per year

#%%
# 初始化日历
start_date = ql.Date(1, 1, 2023)
ql_calendar = QlCalendar(init_date=start_date)
# 创建股票实例
ql_stocks = QlStocks(ql_calendar)
ql_stocks.heston([S0] * n_paths, v0=V0, kappa=kappa, theta=theta, rho=rho, sigma=sigma, dividend_rate=0.0)

#%%
# paths = ql_stocks.create_stock_path_generator(126, paths=n_paths)

new_df = ql_stocks.heston([95.0] * 500, v0=0.04, kappa=1.0, theta=0.06, rho=-0.3, sigma=0.4, dividend_rate=0.01)

#%%
