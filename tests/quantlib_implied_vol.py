import QuantLib as ql
import numpy as np
import time
import matplotlib.pyplot as plt

# 设置基本参数
spot_price = 100.0  # 初始股票价格
strike_price = 100.0
risk_free_rate = 0.05  # 无风险利率
volatility = 0.3  # 波动率
dividend_yield = 0.0  # 股息收益率
maturity = 1.0 / 12  # 时间跨度（1个月，约1/12年）
time_steps = 100  # 时间步数（假设一个月21个交易日）
num_paths = 100  # 模拟路径数量

vol_quote = ql.SimpleQuote(volatility)

# 设置QuantLib日历和日期
calendar = ql.HongKong(ql.HongKong.HKEx)
today = ql.Date(19, 5, 2025)
ql.Settings.instance().evaluationDate = today

# 定义期权参数
day_count = ql.Business252(calendar)
# maturity_date = calendar.advance(today, ql.Period(100, ql.Days), ql.Following)
maturity_date = calendar.advance(today, ql.Period(1, ql.Years), ql.Following)

count = day_count.dayCount(today, maturity_date)
# 设置市场参数
spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
rate_handle = ql.YieldTermStructureHandle(
    ql.FlatForward(today, risk_free_rate, day_count)
)
dividend_handle = ql.YieldTermStructureHandle(
    ql.FlatForward(today, dividend_yield, day_count)
)
vol_handle = ql.BlackVolTermStructureHandle(
    ql.BlackConstantVol(today, calendar, ql.QuoteHandle(vol_quote), day_count)
)

# 设置随机过程（几何布朗运动）
process = ql.BlackScholesMertonProcess(
    spot_handle, dividend_handle, rate_handle, vol_handle
)

#
option_call = ql.EuropeanOption(
    ql.PlainVanillaPayoff(ql.Option.Call, strike_price),
    ql.EuropeanExercise(maturity_date),
)

engine = ql.AnalyticEuropeanEngine(process)
option_call.setPricingEngine(engine)

#
option_put = ql.EuropeanOption(
    ql.PlainVanillaPayoff(ql.Option.Put, strike_price),
    ql.EuropeanExercise(maturity_date),
)

option_put.setPricingEngine(engine)

#
call_npv = option_call.NPV()
call_delta = option_call.delta()
put_npv = option_put.NPV()
put_delta = option_put.delta()

print(call_delta - put_delta)
cash = call_npv * 100 - call_delta * spot_price

print()


# 3. 计算隐含波动率（假设市价为 7.5）
implied_vol = option_call.impliedVolatility(
    10,
    process,
)
print(f"隐含波动率: {implied_vol:.4f}")

vol_quote.setValue(implied_vol)
print(option_call.NPV())

print()