import QuantLib as ql
import numpy as np
import time
import matplotlib.pyplot as plt

# 设置基本参数
spot_price = 100.0  # 初始股票价格
risk_free_rate = 0.05  # 无风险利率
volatility = 0.3  # 波动率
dividend_yield = 0.0  # 股息收益率
maturity = 1.0 / 12  # 时间跨度（1个月，约1/12年）
time_steps = 100  # 时间步数（假设一个月21个交易日）
num_paths = 100  # 模拟路径数量

dividend_yield = ql.SimpleQuote(dividend_yield)
volatility = ql.SimpleQuote(volatility)

# 设置QuantLib日历和日期
calendar = ql.HongKong(ql.HongKong.HKEx)
today = ql.Date(19, 5, 2025)
ql.Settings.instance().evaluationDate = today

# 定义期权参数
day_count = ql.Business252(calendar)
maturity_date = calendar.advance(today, ql.Period(100, ql.Days), ql.Following)

day_count.dayCount(today, maturity_date)
# 设置市场参数
spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
rate_handle = ql.YieldTermStructureHandle(
    ql.FlatForward(today, risk_free_rate, day_count)
)
dividend_handle = ql.YieldTermStructureHandle(
    ql.FlatForward(today, ql.QuoteHandle(dividend_yield), day_count)
)

bc_vol = ql.BlackConstantVol(today, calendar, ql.QuoteHandle(volatility), day_count)

vol_handle = ql.BlackVolTermStructureHandle(
    bc_vol
)

# 设置随机过程（几何布朗运动）
process = ql.BlackScholesMertonProcess(
    spot_handle, dividend_handle, rate_handle, vol_handle
)



# 设置时间网格
times = ql.TimeGrid(maturity, time_steps)

# 设置随机数生成器
rng = ql.GaussianRandomSequenceGenerator(
    ql.UniformRandomSequenceGenerator(
        time_steps, ql.UniformRandomGenerator()
    )
)

# 设置路径生成器
path_generator = ql.GaussianPathGenerator(
    process, maturity, time_steps, rng, False
)
############
np.stack([np.stack(path_generator.next().value()) for _ in range(num_paths)], axis=0)
print('aaaaaaaaaaaaaa')
t = time.time()
aa = [[i for i in path_generator.next().value()] for _ in range(num_paths)]
print(time.time() - t)
t = time.time()
aaa = [np.array(path_generator.next().value()) for _ in range(num_paths)]
print(time.time() - t)
t = time.time()
aaaa = np.array([np.array(path_generator.next().value()) for _ in range(num_paths)])
print('array', time.time() - t)
t = time.time()
aaaaaaaa = np.stack([np.array(path_generator.next().value()) for _ in range(num_paths)], axis=0)
print('array', time.time() - t)
t = time.time()
aaaaa = np.array([list(path_generator.next().value()) for _ in range(num_paths)])
print('array2',time.time() - t)

############
t = time.time()
[i for i in path_generator.next().value()]
print(time.time() - t)
t = time.time()
np.array(path_generator.next().value())
print(time.time() - t)
t = time.time()
list(path_generator.next().value())
print(time.time() - t)
t = time.time()
lll = [i for i in path_generator.next().value()]
print(time.time() - t)

n = [path_generator.next().value() for i in range(5000)]
[path_generator.next().value()[20] for i in range(5000)]

aaaaaaaaaa = [path_generator.next().value() for i in range(3)]

aaaaaaaaaa = np.array([path_generator.next().value() for i in range(3)])

b = [ql.GaussianPathGenerator(
    process, maturity, time_steps, rng, False
) for i in range(5000)]




# 生成多条路径
paths = np.zeros((num_paths, time_steps + 1))
for i in range(num_paths):
    path = path_generator.next().value()
    paths[i, :] = np.array([path[j] for j in range(time_steps + 1)])

# 绘图
plt.figure(figsize=(10, 6))
for i in range(num_paths):
    plt.plot(np.linspace(0, maturity, time_steps + 1), paths[i, :], lw=1)
plt.title("股票价格随机路径模拟（1个月）")
plt.xlabel("时间 (年)")
plt.ylabel("股票价格")
plt.grid(True)
plt.savefig("stock_paths_one_month.png")