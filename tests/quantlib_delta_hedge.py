import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt

# 1. 参数设置
S0 = 100.0  # 初始股价
strike = 100.0  # 行权价
T = 1.0  # 到期时间(年)
r = 0.05  # 无风险利率
sigma = 0.3  # 波动率
dt = 0.01  # 时间步长
M = 500  # 路径步数
num_options = 100  # 期权数量

# 2. 设置QuantLib日期和日历
today = ql.Date().todaysDate()
ql.Settings.instance().evaluationDate = today
calendar = ql.NullCalendar()
day_count = ql.Actual365Fixed()

# 3. 创建市场环境对象
spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
flat_vol = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, sigma, day_count))
process = ql.BlackScholesProcess(spot_handle, flat_ts, flat_vol)


# 4. 生成几何布朗运动路径
def generate_gbm_path(process, T, dt):
    steps = int(T / dt)
    sequence = ql.GaussianRandomSequenceGenerator(
        ql.UniformRandomSequenceGenerator(steps, ql.UniformRandomGenerator()))
    path_generator = ql.GaussianPathGenerator(process, T, steps, sequence, False)
    path = path_generator.next().value()
    return np.array([path[j] for j in range(len(path))])


gbm_path = generate_gbm_path(process, T, dt)
print()

# 5. 期权定价和Delta计算
def bs_option(S, K, t, r, sigma, option_type):
    payoff = ql.PlainVanillaPayoff(option_type, K)
    exercise = ql.EuropeanExercise(today + ql.Period(int(t * 365), ql.Days))
    option = ql.VanillaOption(payoff, exercise)

    engine = ql.AnalyticEuropeanEngine(process)
    option.setPricingEngine(engine)

    # 临时修改标的价格
    spot_quote = spot_handle.currentLink()
    # spot_quote.setValue(S)

    return option.NPV(), option.delta()


# 6. 动态对冲模拟
# 初始化数组
time_points = np.arange(0, T + dt, dt)
portfolio = np.zeros_like(time_points)
cash = np.zeros_like(time_points)
option_values = np.zeros_like(time_points)
deltas = np.zeros_like(time_points)

# 初始计算
option_values[0], deltas[0] = bs_option(S0, strike, T, r, sigma, ql.Option.Call)
cash[0] = option_values[0] * num_options - deltas[0] * S0 * num_options
portfolio[0] = -option_values[0] * num_options + deltas[0] * S0 * num_options

# 动态对冲循环
for t in range(1, len(time_points)):
    current_spot = gbm_path[t]
    remaining_time = T - time_points[t]

    # 更新期权价格和Delta
    option_values[t], deltas[t] = bs_option(
        current_spot, strike, remaining_time, r, sigma, ql.Option.Call)

    # 更新现金账户（利息累积）
    cash[t] = cash[t - 1] * np.exp(r * dt)

    # 更新组合价值
    portfolio[t] = -option_values[t] * num_options + deltas[0] * current_spot * num_options

# 计算PnL
pnl = portfolio + cash

# 7. 可视化结果
plt.figure(figsize=(12, 8))

# 股价路径
plt.subplot(2, 2, 1)
plt.plot(time_points, gbm_path)
plt.title('Stock Price Path')
plt.xlabel('Time')
plt.ylabel('Price')

# 期权价值
plt.subplot(2, 2, 2)
plt.plot(time_points, option_values)
plt.title('Option Value')
plt.xlabel('Time')
plt.ylabel('Value')

# Delta变化
plt.subplot(2, 2, 3)
plt.plot(time_points, deltas)
plt.title('Delta')
plt.xlabel('Time')
plt.ylabel('Delta')

# PnL变化
plt.subplot(2, 2, 4)
plt.plot(time_points, pnl)
plt.title('Portfolio PnL')
plt.xlabel('Time')
plt.ylabel('PnL')

plt.tight_layout()
plt.show()