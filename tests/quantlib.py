import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt
from math import log, sqrt, exp
from scipy.stats import norm
from pylab import mpl

# 设置中文显示
plt.style.use('ggplot')
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# ====================== 债券定价 ======================
def bond_pricing_example():
    # 手动计算债券价格
    f = 100  # 面值
    a = 100 * 0.06 / 2  # 票息，半年一次
    pv = a / (1 + 0.005) ** (0.5) + (a + f) / (1 + 0.007)
    print(f'手动计算债券定价为: {pv:.4f}')

    # 使用QuantLib计算债券价格
    # 定义当前日期(2015年1月15日)
    todaysDate = ql.Date(15, 1, 2015)
    ql.Settings.instance().evaluationDate = todaysDate

    # 构建收益率曲线
    spotDates = [ql.Date(15, 1, 2015), ql.Date(15, 7, 2015), ql.Date(15, 1, 2016)]
    spotRates = [0.0, 0.005, 0.007]
    dayCount = ql.Thirty360()
    calendar = ql.UnitedStates()
    interpolation = ql.Linear()
    compounding = ql.Compounded
    compoundingFrequency = ql.Annual

    spotCurve = ql.ZeroCurve(spotDates, spotRates, dayCount, calendar,
                             interpolation, compounding, compoundingFrequency)
    spotCurveHandle = ql.YieldTermStructureHandle(spotCurve)

    # 构建固定利率债券
    issueDate = ql.Date(15, 1, 2015)
    maturityDate = ql.Date(15, 1, 2016)
    tenor = ql.Period(ql.Semiannual)
    calendar = ql.UnitedStates()
    bussinessConvention = ql.Unadjusted
    dateGeneration = ql.DateGeneration.Backward
    monthEnd = False

    schedule = ql.Schedule(issueDate, maturityDate, tenor, calendar,
                           bussinessConvention, bussinessConvention,
                           dateGeneration, monthEnd)

    dayCount = ql.Thirty360()
    couponRate = 0.06
    coupons = [couponRate]

    settlementDays = 0
    faceValue = 100
    fixedRateBond = ql.FixedRateBond(settlementDays, faceValue, schedule,
                                     coupons, dayCount)

    # 创建债券定价引擎
    bondEngine = ql.DiscountingBondEngine(spotCurveHandle)
    fixedRateBond.setPricingEngine(bondEngine)

    print(f'QuantLib计算债券定价为: {fixedRateBond.NPV():.4f}')

# Black-Scholes公式计算期权价格
def BSM(S0, E, T, r, sigma):
    d1 = (log(S0 / E) + (r + 0.5 * sigma ** 2) * T) / sigma / sqrt(T)
    d2 = d1 - sigma * sqrt(T)
    Callprice = S0 * norm.cdf(d1) - E * exp(-r * T) * norm.cdf(d2)
    print("BS公式计算看涨期权价格: %.4f" % Callprice)



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

    # 日期变更（自动处理节假日）
    new_date = calendar.adjust(ql.Date(18, 6, 2021))
    ql.Settings.instance().evaluationDate = new_date

    # 重新计算剩余期限
    new_T = day_counter.yearFraction(new_date, maturity_date)
    print(f"\n变更后日期: {new_date.to_date()}")
    print(f"实际剩余期限: {new_T:.4f} 年")
    print(f"调整后定价: {option.NPV():.4f}")

    BSM(S0, K, new_T, r, sigma)

    # 计算希腊字母
    print("\n希腊字母:")
    print("%-12s: %4.4f" % ("Delta", option.delta()))
    print("%-12s: %4.4f" % ("Gamma", option.gamma()))
    print("%-12s: %4.4f" % ("Vega", option.vega()))

    return


calculate_option_with_calendar()

def option_pricing_example():
    # 参数
    S0 = 100.0
    E = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.20

    BSM(S0, E, T, r, sigma)

    # 使用QuantLib计算期权价格
    # 设定全局估值日
    today = ql.Date(18, 11, 2020)
    ql.Settings.instance().evaluationDate = today

    # 构建期权
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, 100.0)
    europeanExercise = ql.EuropeanExercise(ql.Date(18, 11, 2021))
    option = ql.EuropeanOption(payoff, europeanExercise)

    # 输入参数
    u = ql.SimpleQuote(100.0)  # 标的资产价值
    r = ql.SimpleQuote(0.05)  # 无风险利率
    sigma = ql.SimpleQuote(0.20)  # 波动率

    # 构建收益率曲线和波动率曲线
    riskFreeCurve = ql.FlatForward(0, ql.TARGET(), ql.QuoteHandle(r), ql.Actual360())
    volatility = ql.BlackConstantVol(0, ql.TARGET(), ql.QuoteHandle(sigma), ql.Actual360())

    # 初始化BS过程，并构造engine
    process = ql.BlackScholesProcess(ql.QuoteHandle(u),
                                     ql.YieldTermStructureHandle(riskFreeCurve),
                                     ql.BlackVolTermStructureHandle(volatility))
    engine = ql.AnalyticEuropeanEngine(process)

    # 对期权设定该engine
    option.setPricingEngine(engine)
    print(f'QuantLib计算看涨期权价格: {option.NPV():.4f}')

    ql.Settings.instance().evaluationDate = ql.Date(18, 6, 2021)
    print(f'日期变更: {option.NPV():.4f}')

    # 计算希腊字母
    print("\n希腊字母:")
    print("%-12s: %4.4f" % ("Delta", option.delta()))
    print("%-12s: %4.4f" % ("Gamma", option.gamma()))
    print("%-12s: %4.4f" % ("Vega", option.vega()))

    # 改变参数重新计算
    print("\n改变参数后的期权价格:")
    u.setValue(105.0)
    print(f'标的资产=105时: {option.NPV():.4f}')

    u.setValue(98.0)
    r.setValue(0.04)
    sigma.setValue(0.25)
    print(f'标的资产=98, 利率=4%, 波动率=25%时: {option.NPV():.4f}')

    ql.Settings.instance().evaluationDate = ql.Date(18, 6, 2021)
    print(f'日期变更: {option.NPV():.4f}')


    # 绘制期权价值与标的资产价格关系图
    X = np.linspace(80.0, 120.0, 400)
    cv = []
    for i in X:
        u.setValue(i)
        cv.append(option.NPV())

    plt.figure(figsize=(10, 6))
    plt.plot(X, cv, linewidth=2)
    plt.title('期权价值-标的资产价值', size=15)
    plt.xlabel('标的资产价格')
    plt.ylabel('期权价格')
    plt.show()

    # 不同估值日期对比
    ql.Settings.instance().evaluationDate = ql.Date(18, 6, 2021)
    y = []
    for i in X:
        u.setValue(i)
        y.append(option.NPV())

    plt.figure(figsize=(10, 6))
    plt.plot(X, y, '--', linewidth=2, color='b', label='估值日:2021.6.18')
    plt.plot(X, cv, linewidth=2, color='r', label='估值日:2020.11.18')
    plt.legend()
    plt.title('不同估值日下期权价值-标的资产价值', size=15)
    plt.xlabel('标的资产价格')
    plt.ylabel('期权价格')
    plt.show()


# 执行示例
print("============= 债券定价示例 =============")
# bond_pricing_example()

print("\n============= 期权定价示例 =============")
option_pricing_example()