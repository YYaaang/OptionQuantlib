# %% md
# QuantLib 期权定价示例
#### 本示例展示如何使用`QlCalendar`、`QlStocks`和`QlVanillaOptions`类创建和定价多种期权，并测试不同定价引擎的功能。

# %%
import QuantLib as ql
import numpy as np
import pandas as pd
from src.QlCalendar import QlCalendar
from src.QlStocks import QlStocks
from src.QlVanillaOptions import QlVanillaOptions

# %% md
## 1. 初始化设置
#### 创建日历、股票和期权对象
# %%
# 设置初始日期
start_date = ql.Date(31, 5, 2025)  # 2025年5月31日
ql_calendar = QlCalendar(init_date=start_date)

# 创建股票对象
s = QlStocks(ql_calendar)
s.black_scholes(100.0)  # 创建一只股票，价格为100

# 创建期权对象
options = QlVanillaOptions(s.df.loc[0])  # 使用第一只股票创建期权

# %% md
## 2. 测试不同定价引擎
### 2.1 AnalyticEuropeanEngine (解析解)
# %%
# 创建单个欧式看跌期权
op = options.european_option(
    'put',
    100,  # 行权价
    ql.Date(30, 6, 2025),  # 到期日
    qlEngine=ql.AnalyticEuropeanEngine
)

# 计算并显示结果
df = options.NPV(op)
print("AnalyticEuropeanEngine 结果:")
print(df[['strike_prices', 'maturity_dates', 'types', 'NPV']])

# %% md
### 2.2 MCEuropeanEngine (蒙特卡洛模拟)
# %%
# 创建多个欧式看涨期权
strikes = np.linspace(90, 110, 5)  # 5个不同行权价
maturities = [ql.Date(30, 6, 2025)] * len(strikes)  # 相同到期日
types = ['call'] * len(strikes)  # 全部为看涨期权

op = options.european_option(
    types,
    strikes,
    maturities,
    qlEngine=ql.MCEuropeanEngine,
    traits = "pseudorandom", # could use "lowdiscrepancy"
    timeSteps=20,  # 时间步数
    requiredSamples=1000,  # 路径数量
    seed=42  # 随机种子
)

# 计算并显示结果
df = options.NPV(op)
print("\nMCEuropeanEngine 结果:")
print(df[['strike_prices', 'maturity_dates', 'types', 'NPV']])

# %% md
### 2.3 FdBlackScholesVanillaEngine (有限差分法)
# %%
# 创建多个欧式看跌期权
strikes = np.linspace(80, 120, 5)  # 5个不同行权价
maturities = [ql.Date(31, 12, 2025)] * len(strikes)  # 相同到期日
types = ['put'] * len(strikes)  # 全部为看跌期权

tGrid, xGrid = 2000, 200
op = options.european_option(
    types,
    strikes,
    maturities,
    ql.EuropeanExercise,
    ql.FdBlackScholesVanillaEngine,
    tGrid,
    xGrid
)

# 计算并显示结果
df = options.NPV(op)
print("\nFdBlackScholesVanillaEngine 结果:")
print(df[['strike_prices', 'maturity_dates', 'types', 'NPV']])

# %% md
### 2.4 MCAmericanEngine (美式期权蒙特卡洛)
# %%
# 创建多个美式看涨期权
# strikes = np.linspace(95, 105, 3)  # 3个不同行权价
# maturities = [ql.Date(31, 12, 2025)] * len(strikes)  # 相同到期日
# types = ['call'] * len(strikes)  # 全部为看涨期权
#
# op = options.european_option(
#     types,
#     strikes,
#     maturities,
#     qlExercise=ql.AmericanExercise,
#     qlEngine=ql.MCAmericanEngine,
#     timeSteps = 200,
#     traits = "pseudorandom", # could use "lowdiscrepancy"
#     requiredSamples=1000,  # 路径数量
#     seed=42  # 随机种子
# )
#
# # 计算并显示结果
# df = options.NPV(op)
# print("\nMCAmericanEngine 结果:")
# print(df[['strike_prices', 'maturity_dates', 'types', 'NPV']])
#
# # %% md
# ## 3. 测试风险指标计算
# #### 计算并显示所有期权的希腊字母
# # %%
# # 计算delta
# df_delta = options.delta()
# print("\nDelta 值:")
# print(df_delta[['strike_prices', 'types', 'delta']])
#
# # 计算gamma
# df_gamma = options.gamma()
# print("\nGamma 值:")
# print(df_gamma[['strike_prices', 'types', 'gamma']])
#
# # 计算vega
# df_vega = options.vega()
# print("\nVega 值:")
# print(df_vega[['strike_prices', 'types', 'vega']])
#
# # %% md
# ## 4. 批量创建大量期权
# #### 测试5000个期权的批量创建和定价
# # %%
# # 生成随机行权价
# prices = np.random.normal(loc=100, scale=10, size=5000)
#
# # 创建5000个看涨期权
# options.european_option(
#     'call',
#     prices,
#     ql.Date(31, 12, 2025),
#     qlEngine=ql.AnalyticEuropeanEngine
# )
#
# # 计算并显示部分结果
# df = options.NPV()
# print("\n5000个期权的部分结果:")
# print(df.head())  # 只显示前5行
#
# # 计算并显示统计信息
# print("\n5000个期权的统计信息:")
# print(df['NPV'].describe())