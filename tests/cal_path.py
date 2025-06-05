import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt


def generate_price_path():
    # 参数设置
    S0 = 100.0  # 初始股价
    strike = 100.0  # 行权价
    r = 0.05  # 无风险利率
    sigma = 0.20  # 波动率
    maturity = 1.0  # 到期时间（年）
    steps = 252  # 时间步数（假设252个交易日）
    n_paths = 1  # 生成1条路径

    # 设置估值日期
    today = ql.Date(18, 11, 2020)
    ql.Settings.instance().evaluationDate = today

    # 创建随机过程（几何布朗运动）
    riskFreeCurve = ql.FlatForward(0, ql.TARGET(), r, ql.Actual360())
    volatility = ql.BlackConstantVol(0, ql.TARGET(), sigma, ql.Actual360())
    process = ql.BlackScholesProcess(
        ql.QuoteHandle(ql.SimpleQuote(S0)),
        ql.YieldTermStructureHandle(riskFreeCurve),
        ql.BlackVolTermStructureHandle(volatility))

    # 创建随机数生成器
    rng = 42
    urng = ql.UniformRandomGenerator(rng)
    gaussian = ql.GaussianRandomGenerator(urng)

    # 创建路径生成器
    sequenceGenerator = ql.GaussianRandomSequenceGenerator(
        ql.UniformRandomSequenceGenerator(steps, urng))
    pathGenerator = ql.GaussianPathGenerator(
        process, maturity, steps, sequenceGenerator, False)

    # 生成路径
    path = pathGenerator.next().value()
    time = [x for x in path.time()]
    prices = [path[x] for x in range(len(path))]

    return time, prices


def calculate_option_prices_along_path(time, prices):
    # 期权参数
    strike = 100.0
    r = 0.05
    sigma = 0.20
    maturity_date = ql.Date(18, 11, 2021)

    # 存储期权价格
    option_prices = []

    # 构建期权
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
    exercise = ql.EuropeanExercise(maturity_date)
    option = ql.EuropeanOption(payoff, exercise)

    # 对路径上的每个点计算期权价格
    for i, (t, S) in enumerate(zip(time, prices)):
        # 设置当前日期
        current_date = ql.Date(18, 11, 2020) + ql.Period(int(t * 365), ql.Days)
        ql.Settings.instance().evaluationDate = current_date

        # 剩余时间
        remaining_time = time[-1] - t

        # 更新过程参数
        u = ql.SimpleQuote(S)
        riskFreeCurve = ql.FlatForward(0, ql.TARGET(), r, ql.Actual360())
        volatility = ql.BlackConstantVol(0, ql.TARGET(), sigma, ql.Actual360())

        process = ql.BlackScholesProcess(
            ql.QuoteHandle(u),
            ql.YieldTermStructureHandle(riskFreeCurve),
            ql.BlackVolTermStructureHandle(volatility))

        engine = ql.AnalyticEuropeanEngine(process)
        option.setPricingEngine(engine)

        option_prices.append(option.NPV())

    return option_prices


# 生成路径并计算期权价格
time, prices = generate_price_path()
option_prices = calculate_option_prices_along_path(time, prices)

# 绘制结果
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time, prices, 'b-', label='Stock Price')
plt.title('Stock Price Path')
plt.ylabel('Price')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, option_prices, 'r-', label='Option Price')
plt.title('Option Price Along Path')
plt.xlabel('Time (years)')
plt.ylabel('Option Price')
plt.legend()

plt.tight_layout()
plt.show()