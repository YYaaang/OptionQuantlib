import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt
from math import log, sqrt, exp
from scipy.stats import norm
from pylab import mpl

# Black-Scholes公式计算期权价格
def BSM(S0, E, T, r, sigma):
    d1 = (log(S0 / E) + (r + 0.5 * sigma ** 2) * T) / sigma / sqrt(T)
    d2 = d1 - sigma * sqrt(T)
    Callprice = S0 * norm.cdf(d1) - E * exp(-r * T) * norm.cdf(d2)
    print("BS公式计算看涨期权价格: %.4f" % Callprice)

def bs_delta(S0, strike, T, r, sigma, type):
    d1 = (np.log(S0/strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    data = norm.cdf(d1)
    if type == 'call':
        print(f"BS公式计算{type} delta: {data}")
        return data
    elif type == 'put':
        print(f"BS公式计算{type} delta: {data - 1}")
        return data - 1
    else:
        raise ValueError('Option type must be either "call" or "put"')


# ====================== 期权定价 ======================
def calculate_option_with_calendar():
    # 参数
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.20


    # 显式选择市场日历（香港市场示例）
    calendar = ql.HongKong(ql.HongKong.HKEx)

    # 日期设置（自动调整到最近交易日）
    today = calendar.adjust(ql.Date(18, 11, 2020))
    ql.Settings.instance().evaluationDate = today

    # 到期日设置（需检查是否为交易日）
    maturity_date = calendar.adjust(ql.Date(18, 11, 2021))

    # 计算实际剩余时间（考虑交易日历）
    day_counter = ql.Actual365Fixed()
    T = day_counter.yearFraction(today, maturity_date)  # 按交易日计算实际年化时间

    BSM(S0, K, T, r, sigma)


    # 构建期权
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    exercise = ql.EuropeanExercise(maturity_date)
    option = ql.EuropeanOption(payoff, exercise)

    # 市场参数（传入日历）
    u = ql.SimpleQuote(S0)
    quote_handle = ql.QuoteHandle(u)
    yield_curve = ql.FlatForward(0, calendar, ql.QuoteHandle(ql.SimpleQuote(r)), day_counter)
    vol_curve = ql.BlackConstantVol(0, calendar, ql.QuoteHandle(ql.SimpleQuote(sigma)), day_counter)

    # 随机过程
    process = ql.BlackScholesProcess(quote_handle,
                                     ql.YieldTermStructureHandle(yield_curve),
                                     ql.BlackVolTermStructureHandle(vol_curve))

    # 定价引擎
    engine = ql.AnalyticEuropeanEngine(process)
    option.setPricingEngine(engine)

    # 初始定价
    print(f"初始定价(2020-11-18): {option.NPV():.4f}")
    print(f"理论剩余期限: {T:.4f} 年")

    # 计算希腊字母
    print("\n希腊字母:")
    print("%-12s: %4.4f" % ("Delta", option.delta()))
    print("%-12s: %4.4f" % ("Gamma", option.gamma()))
    print("%-12s: %4.4f" % ("Vega", option.vega()))

    bs_delta(S0, K, T, r, sigma, 'call')

    # 日期变更（自动处理节假日）
    new_date = calendar.adjust(ql.Date(18, 6, 2021))
    ql.Settings.instance().evaluationDate = new_date
    new_S0 = 105.0
    u.setValue(new_S0)

    # 重新计算剩余期限
    new_T = day_counter.yearFraction(new_date, maturity_date)
    print(f"\n变更后日期: {new_date.to_date()}, \n变更后S0: {new_S0}")
    print(f"实际剩余期限: {new_T:.4f} 年")
    print(f"调整后定价: {option.NPV():.4f}")

    BSM(new_S0, K, new_T, r, sigma)


    return


calculate_option_with_calendar()
