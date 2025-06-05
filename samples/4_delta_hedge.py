#%%
import sys
import os
current_path = os.getcwd()  # 获取当前工作目录
print("当前路径:", current_path)
current_path = current_path.replace('/samples', '')
print(sys.path)  # 显示所有模块搜索路径
sys.path.append(current_path)  # 添加自定义路径
print(sys.path)  # 显示所有模块搜索路径

#%%
import numpy as np
import QuantLib as ql

from src.QlCalendar import QlCalendar
from src.QlStocks import QlStocks
from src.QlVanillaOptions import QlVanillaOptions

#%%
S0 = 100
strike = 100
r = 0.03
T=1
paths = 1000
steps = 252
#%%
start_date = ql.Date(1, 1, 2023)
ql_calendar = QlCalendar(init_date=start_date, init_risk_free_rate=r)
end_date = ql_calendar.cal_date_advance(times=T, time_unit='years')
#%%
stock_1 = QlStocks(ql_calendar)
stock_1.black_scholes([S0] * paths, sigma=0.2)
#%%
options = QlVanillaOptions(stock_1.df)
options.european_option(
    'call',
    strike,
    ql.Date(1, 1, 2024)
)
#%%
# # test if fulfills put-call parity

#%%
# print(options.NPV() - option_put.NPV() - (S0 - strike * np.exp(-r * T)))

npv = options.NPV()
delta = options.delta()

#%%
path_list = stock_1.stock_path_generator(steps, paths=paths)

#%%
# 动态对冲模拟
# 初始化数组
dt = T / steps
time_points = np.arange(0, T + dt, dt)
portfolio = np.zeros((steps+1, 1000))
cash = np.zeros((steps+1, 1000))
option_values = np.zeros((steps+1, 1000))
deltas = np.zeros((steps+1, 1000))
#%%
# 初始计算
option_values[0] = options.NPV()['NPV']
deltas[0] = options.delta()['delta']
cash[0] = option_values[0] - deltas[0] * S0
portfolio[0] = -option_values[0] + deltas[0] * S0

#%%
all_trade_dates = [ql_calendar.today]
for t in range(1, steps):
    # 设置价格
    ql_calendar.to_next_trading_date()
    print(ql_calendar.today)
    all_trade_dates.append(ql_calendar.today)
    stock_1.set_prices([p[t] for p in path_list])
    #
    option_values[t] = options.NPV()['NPV']
    deltas[t] = options.delta()['delta']
    cash[t] = cash[t - 1] * np.exp(r * dt)
    portfolio[t] = -option_values[t] + deltas[t] * S0

#%%
# 计算PnL
pnl = portfolio + cash
#%%

# 可视化结果
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

# 股价路径
plt.subplot(2, 2, 1)
plt.plot(time_points, np.array(path_list).T)
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
#%%
print()
