import QuantLib as ql
import numpy as np
import pandas as pd
import time

S0 = 100
strike = 100
r = 0.05
sigma = 0.3
T=1

# 1. 输入参数设置
stock_prices = np.random.normal(loc=100, scale=1, size=10000)  # 示例股票价格列表
risk_free_rate = 0.05  # 年化无风险利率(5%)
volatility = 0.20  # 年化波动率(20%)
calculation_date = ql.Date(26, 5, 2025)  # 当前日期
ql.Settings.instance().evaluationDate = calculation_date

# 2. 创建期限结构
day_count = ql.Actual365Fixed()
risk_free_curve = ql.YieldTermStructureHandle(
    ql.FlatForward(calculation_date, risk_free_rate, day_count))
volatility_curve = ql.BlackVolTermStructureHandle(
    ql.BlackConstantVol(calculation_date, ql.NullCalendar(), volatility, day_count))

# 3. 为每个股票价格创建BlackScholesProcess
processes = []
stock_prices_quote = [ql.SimpleQuote(i) for i in stock_prices]
for price in stock_prices_quote:
    # 创建报价对象
    spot_handle = ql.QuoteHandle(price)

    # 创建过程对象（无股息版本）
    process = ql.BlackScholesProcess(
        spot_handle,
        risk_free_curve,
        volatility_curve
    )
    processes.append(process)

# 4. 验证过程创建
print(f"成功创建{len(processes)}个BlackScholesProcess实例")
print("第一个过程的参数验证：")
print(f"初始价格: {processes[0].x0()}")
print(f"无风险利率: {processes[0].riskFreeRate().zeroRate(1.0, ql.Continuous).rate():.2%}")
print(f"波动率: {processes[0].blackVolatility().blackVol(1.0, processes[0].x0()):.2%}")


payoff = ql.PlainVanillaPayoff(ql.Option.Call, 100)
exercise = ql.EuropeanExercise(calculation_date + ql.Period(3, ql.Months))

options = []
for i in processes:
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(i))
    options.append(option)

current_prices = [i.NPV() for i in options]
current_delta = [i.delta() for i in options]

#
t = time.time()

stock_prices_new = np.random.normal(loc=100, scale=1, size=10000)  # 示例股票价格列表
for i in range(len(stock_prices_quote)):
    stock_prices_quote[i].setValue(stock_prices_new[i])
current_prices1 = [i.NPV() for i in options]
current_delta1 = [i.delta() for i in options]

print(time.time() - t)

print()