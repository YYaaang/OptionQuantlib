import QuantLib as ql
import numpy as np
import timeit
import time

# 基础参数
spot_price = 100.0
strike = 100.0
volatility = ql.SimpleQuote(0.2)
risk_free_rate = 0.05
maturity = 1.0  # 1年
calculation_date = ql.Date().todaysDate()
ql.Settings.instance().evaluationDate = calculation_date
#
sport_quote = ql.SimpleQuote(spot_price)
# 构建期权对象
payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
exercise = ql.EuropeanExercise(calculation_date + ql.Period(int(maturity * 365), ql.Days))
option = ql.VanillaOption(payoff, exercise)

# 配置定价引擎
riskFreeCurve = ql.FlatForward(calculation_date, risk_free_rate, ql.Actual365Fixed())
volatilityCurve = ql.BlackConstantVol(calculation_date, ql.TARGET(), ql.QuoteHandle(volatility), ql.Actual365Fixed())
dividend_curve_handle =  ql.YieldTermStructureHandle(
    ql.FlatForward(calculation_date, 0.02, ql.Actual365Fixed())
    )
process = ql.BlackScholesMertonProcess(
    ql.QuoteHandle(sport_quote),
    dividend_curve_handle,
    ql.YieldTermStructureHandle(riskFreeCurve),
    ql.BlackVolTermStructureHandle(volatilityCurve)
)

option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
print(option.NPV())
sport_quote.setValue(110)
print(option.NPV())
print(option.impliedVolatility(15, process))
#%%
# time
print('npv 10000', timeit.timeit(option.NPV, number=10000))
#%%
# 查看set Engine 时间
def set_engine():
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(sport_quote),
        dividend_curve_handle,
        ql.YieldTermStructureHandle(riskFreeCurve),
        ql.BlackVolTermStructureHandle(volatilityCurve)
    )
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return
print('set_engine 10000', timeit.timeit(set_engine, number=10000))
#%%
#
def set_vol():
    volatility.setValue(0.3)
print('set_vol 10000', timeit.timeit(set_vol, number=10000))
#%%
# def next_dates():
#     def next_date():
#         ql.Settings.instance().evaluationDate += 1
#     n = 365
#     [next_date() for _ in range(n)]
# # print(ql.Settings.instance().evaluationDate)
# print('change_date', timeit.timeit(next_dates, number=1))
# # print(ql.Settings.instance().evaluationDate)
#%%
# a = {'a': 5}
# b = None
# n = 10000 * 365
# def get_prices():
#     if b is None:
#         a = 5
# print('change_prices', timeit.timeit(get_prices, number=n))
#%%
n = 10000
def get_prices():
    option.impliedVolatility(15, process)
    # sport_quote.value()
print('change_prices ppp', timeit.timeit(get_prices, number=n))
#%%
n = 10000
prices = np.random.normal(loc=100.0, scale=10.0, size=n)
def new_prices():
    def next_prices(i):
        sport_quote.setValue(prices[i])
        # print(prices[i], sport_quote.value())
        return
    [next_prices(i) for i in range(n)]
print('change_prices', timeit.timeit(new_prices, number=1))
#%%
n = 100000
prices = np.random.normal(loc=100.0, scale=10.0, size=n)
def cal_NPV(price):
    sport_quote.setValue(price)
    return option.NPV()

def new_NPVs():
    return np.array([cal_NPV(i) for i in prices])
print('1 change_prices and get NPV', timeit.timeit(new_NPVs, number=10))
def new_vectorize_NPVs():
    return np.vectorize(cal_NPV)(prices)
print('2 change_prices and get NPV', timeit.timeit(new_vectorize_NPVs, number=10))
# print(prices[-1], sport_quote.value())
#%%
ql.Settings.instance().evaluationDate = calculation_date
# print(ql.Settings.instance().evaluationDate, sport_quote.value())
prices = np.random.normal(loc=100.0, scale=10.0, size=365)
def new_dates_prices():
    def next_date_prices(i):
        ql.Settings.instance().evaluationDate += 1
        sport_quote.setValue(prices[i])
        return
    n = 365
    [next_date_prices(i) for i in range(n)]
print('change dates and prices', timeit.timeit(new_dates_prices, number=1))
# print(ql.Settings.instance().evaluationDate, sport_quote.value())
#%%
ql.Settings.instance().evaluationDate = calculation_date

ql.Settings.instance().evaluationDate = calculation_date
# print(ql.Settings.instance().evaluationDate, sport_quote.value())
prices = np.random.normal(loc=100.0, scale=10.0, size=[365, 10000])
def new_dates_prices():
    def next_date_prices(i):
        ql.Settings.instance().evaluationDate += 1
        [sport_quote.setValue(prices[i, j]) for j in range(prices.shape[1])]
        return
    [next_date_prices(i) for i in range(prices.shape[0])]
print('change 365 dates,10000 prices per date', timeit.timeit(new_dates_prices, number=1))
# print(ql.Settings.instance().evaluationDate, prices[-1,-1], sport_quote.value())
#%%
length = 1
timesteps = 252
urng = ql.UniformRandomGenerator(24)
sequenceGenerator = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(timesteps, urng))
pathGenerator = ql.GaussianPathGenerator(
    process, length, timesteps, sequenceGenerator, False)
def generate():
    return np.array(pathGenerator.next().value())
print('generate paths', timeit.timeit(generate, number=10000))
#
def multi_generate():
    return [generate() for i in range(10000)]
#
print('generate multi paths', timeit.timeit(multi_generate, number=1))
t0 = time.time()
[np.array(pathGenerator.next().value()) for i in range(10000)]
print(time.time() - t0)
print()