import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 设置随机种子以确保可重现性
np.random.seed(13337)

# 参数设置
S0 = 100.0         # 初始股票价格
strike = 100.0     # 期权行权价
T = 1.0            # 期权到期时间（年）
r = 0.05           # 无风险利率
sigma = 0.3        # 波动率
dt = 0.01          # 时间步长
M = 500            # 模拟路径数量
n_steps = int(T / dt)  # 时间步数

# 设置QuantLib全局评估日期
today = ql.Date(15, 5, 2025)
ql.Settings.instance().evaluationDate = today

# 定义日历和日计数惯例
calendar = ql.HongKong(ql.HongKong.HKEx)
day_count = ql.Actual365Fixed()

# 定义结算日期（假设为即期交易，2个工作日后）
settlement_date = calendar.advance(today, 2, ql.Days)

# 定义无风险利率曲线和股息曲线
risk_free_rate = ql.FlatForward(settlement_date, r, day_count)
risk_free_curve = ql.YieldTermStructureHandle(risk_free_rate)
dividend_rate = ql.FlatForward(settlement_date, 0.0, day_count)
dividend_curve = ql.YieldTermStructureHandle(dividend_rate)

# 定义波动率曲线
volatility = ql.BlackConstantVol(settlement_date, calendar, sigma, day_count)
volatility_curve = ql.BlackVolTermStructureHandle(volatility)

# 定义标的资产价格句柄
spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))

# 创建Black-Scholes-Merton过程
bsm_process = ql.BlackScholesMertonProcess(
    spot_handle, dividend_curve, risk_free_curve, volatility_curve
)

# 定义期权
option_type = ql.Option.Call
payoff = ql.PlainVanillaPayoff(option_type, strike)
exercise = ql.EuropeanExercise(calendar.advance(today, int(T * 252), ql.Days))
option = ql.VanillaOption(payoff, exercise)

# 设置定价引擎
engine = ql.AnalyticEuropeanEngine(bsm_process)
option.setPricingEngine(engine)

# 生成股票价格路径
times = np.linspace(0, T, n_steps + 1)
time_grid = list(times)
rng = ql.GaussianRandomSequenceGenerator(
    ql.UniformRandomSequenceGenerator(n_steps, ql.UniformRandomGenerator(1234))
)
path_generator = ql.GaussianMultiPathGenerator(bsm_process, time_grid, rng, False)

stock_paths = np.zeros((n_steps + 1, M))
for i in range(M):
    path = path_generator.next().value()[0]  # 提取股票价格路径
    stock_paths[:, i] = [path[j] for j in range(n_steps + 1)]

# Delta对冲模拟
option_paths = np.zeros((n_steps + 1, M))
option_deltas = np.zeros((n_steps + 1, M))
portfolios = np.zeros((n_steps + 1, M))
cash = np.zeros((n_steps + 1, M))
pnl = np.zeros((n_steps + 1, M))

# 初始期权价格和Delta
initial_price = option.NPV()
option_paths[0, :] = initial_price
initial_delta = option.delta()
option_deltas[0, :] = initial_delta

# 初始现金账户和投资组合价值
cash[0, :] = initial_price * 100 - initial_delta * S0
portfolios[0, :] = -initial_price * 100 + initial_delta * S0

# 遍历时间步
for t in range(1, n_steps + 1):
    time_to_maturity = T - t * dt
    if time_to_maturity <= 0:
        time_to_maturity = 1e-10  # 防止到期时除零
    for i in range(M):
        stock_price = stock_paths[t, i]
        # 更新标的资产价格
        spot_handle.value = stock_price
        # 更新到期时间
        exercise_date = calendar.advance(today, int(time_to_maturity * 252), ql.Days)
        option.exercise = ql.EuropeanExercise(exercise_date)
        # 计算期权价格和Delta
        option_price = option.NPV()
        option_delta = option.delta()
        option_paths[t, i] = option_price
        option_deltas[t, i] = option_delta
        # 更新投资组合价值
        portfolios[t, i] = -option_price * 100 + initial_delta * stock_price
        # 更新现金账户
        cash[t, i] = cash[t-1, i] * np.exp(r * dt)
        # 更新PnL
        pnl[t, i] = portfolios[t, i] + cash[t, i]

# 可视化
fig, ax = plt.subplots(2, 3, figsize=(18, 8))

# 股票价格
ax[0, 0].plot(stock_paths)
ax[0, 0].set_title('Stock Price')

# 期权价格
ax[0, 1].plot(option_paths)
ax[0, 1].set_title('Option Price')

# 期权Delta
ax[0, 2].plot(option_deltas)
ax[0, 2].set_title('Option Delta')

# 投资组合价值
ax[1, 0].plot(portfolios)
ax[1, 0].set_title('Portfolio Value')

# 现金账户
ax[1, 1].plot(cash)
ax[1, 1].set_title('Cash Account')

# PnL
ax[1, 2].plot(pnl)
ax[1, 2].set_title('PnL')

plt.tight_layout()
plt.show()
plt.close()

# 最终PnL分布
plt.hist(pnl[-1, :], bins=50)
plt.title('Final PnL Distribution')
plt.show()
plt.close()