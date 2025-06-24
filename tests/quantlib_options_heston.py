import QuantLib as ql

# 设置基本参数
spot_price = 100.0  # 初始股票价格
strike_price = 100.0
risk_free_rate = 0.05  # 无风险利率
volatility = 0.3  # 波动率
dividend_yield = 0.0  # 股息收益率
maturity = 1.0 / 12  # 时间跨度（1个月，约1/12年）
time_steps = 100  # 时间步数（假设一个月21个交易日）
num_paths = 100  # 模拟路径数量

V0 = 0.04  # Initial volatility
kappa = 2.0  # Mean reversion speed of volatility
theta = 0.04  # Long-term mean of volatility
sigma = 0.3  # Volatility of volatility
rho = -0.7  # Correlation between price and volatility
r = 0.05  # Risk-free rate
T = 1.0  # Option maturity (years)
K = 100.0  # Option strike price
trading_days = 252  # Trading days per year

# 设置QuantLib日历和日期
calendar = ql.HongKong(ql.HongKong.HKEx)
today = ql.Date(3, 1, 2023)
ql.Settings.instance().evaluationDate = today

# 定义期权参数
day_count = ql.Business252(calendar)
# maturity_date = calendar.advance(today, ql.Period(100, ql.Days), ql.Following)
maturity_date = calendar.advance(today, ql.Period(trading_days, ql.Days), ql.Following)

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
    ql.BlackConstantVol(today, calendar, volatility, day_count)
)

# 设置随机过程（几何布朗运动）
hestonProcess = ql.HestonProcess(
    rate_handle, dividend_handle, spot_handle,
    V0, kappa, theta, sigma, rho
)
hestonModel = ql.HestonModel(hestonProcess)
engine = ql.AnalyticHestonEngine(hestonModel)
#
option_call = ql.EuropeanOption(
    ql.PlainVanillaPayoff(ql.Option.Call, strike_price),
    ql.EuropeanExercise(maturity_date),
)

option_call.setPricingEngine(engine)

#
option_put = ql.EuropeanOption(
    ql.PlainVanillaPayoff(ql.Option.Put, strike_price),
    ql.EuropeanExercise(maturity_date),
)

option_put.setPricingEngine(engine)

#
call_npv = option_call.NPV()
put_npv = option_put.NPV()
call_delta = option_call.delta()
put_delta = option_put.delta()

print(call_delta - put_delta)
cash = call_npv * 100 - call_delta * spot_price


print()
