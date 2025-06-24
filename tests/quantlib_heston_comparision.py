import QuantLib as ql
import time
import pandas as pd

# 设置全局评估日期
# calculation_date = ql.Date(9, 6, 2025)
# ql.Settings.instance().evaluationDate = calculation_date

# 参数设置
S0 = 100.0  # 初始股票价格
V0 = 0.04  # 初始方差
kappa = 2.0  # 均值回归速度
theta = 0.04  # 长期方差均值
sigma = 0.3  # 波动率的波动率
rho = -0.7  # 相关系数
r = 0.05  # 无风险利率
T = 1.0  # 到期时间（年）
K = 100.0  # 执行价格
trading_days = 252  # 每年交易日数


# 设置QuantLib日历和日期
calendar = ql.HongKong(ql.HongKong.HKEx)
today = ql.Date(3, 1, 2023)
ql.Settings.instance().evaluationDate = today
day_count = ql.Business252(calendar)
maturity_date = calendar.advance(today, ql.Period(trading_days, ql.Days), ql.Following)

# 构造期权
# maturity_date = calculation_date + int(T * trading_days)
payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
exercise = ql.EuropeanExercise(maturity_date)
european_option = ql.VanillaOption(payoff, exercise)

# 构造 Heston 模型的公共组件
# day_count = ql.Actual365Fixed()
# calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
risk_free_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, day_count))
heston_process = ql.HestonProcess(risk_free_ts, dividend_ts, spot_handle, V0, kappa, theta, sigma, rho)
heston_model = ql.HestonModel(heston_process)

engine = ql.AnalyticHestonEngine(heston_model)
#
option_call = ql.EuropeanOption(
    ql.PlainVanillaPayoff(ql.Option.Call, K),
    ql.EuropeanExercise(maturity_date),
)

option_call.setPricingEngine(engine)

#
option_put = ql.EuropeanOption(
    ql.PlainVanillaPayoff(ql.Option.Put, K),
    ql.EuropeanExercise(maturity_date),
)

option_put.setPricingEngine(engine)

#
call_npv = option_call.NPV()
put_npv = option_put.NPV()



# 定义四种定价引擎
engines = {
    "AnalyticHestonEngine": ql.AnalyticHestonEngine(heston_model),
    "MCEuropeanHestonEngine": ql.MCEuropeanHestonEngine(
        heston_process, "PseudoRandom", timeSteps=252, requiredSamples=1000, seed=42
    ),
    "FdHestonVanillaEngine": ql.FdHestonVanillaEngine(heston_model, 100, 100, 50),
    # "AnalyticPTDHestonEngine": ql.AnalyticPTDHestonEngine(
    #     ql.PiecewiseTimeDependentHestonModel(
    #         risk_free_ts, dividend_ts, spot_handle, V0,
    #         ql.PiecewiseConstantParameter([T], ql.PositiveConstraint(), [theta]),
    #         ql.PiecewiseConstantParameter([T], ql.PositiveConstraint(), [kappa]),
    #         ql.PiecewiseConstantParameter([T], ql.PositiveConstraint(), [sigma]),
    #         ql.PiecewiseConstantParameter([T], ql.BoundaryConstraint(-1.0, 1.0), [rho]),
    #         ql.TimeGrid(T, 10)
    #     )
    # )
}

# 存储结果
results = []

# 计算期权价格和 Delta
for engine_name, engine in engines.items():
    european_option.setPricingEngine(engine)
    start_time = time.time()
    option_price = european_option.NPV()
    elapsed_time = time.time() - start_time

    # 尝试计算 Delta
    delta = None
    try:
        delta = european_option.delta()
    except Exception as e:
        delta = "不可用"

    results.append({
        "引擎": engine_name,
        "期权价格": round(option_price, 6),
        "计算时间 (秒)": round(elapsed_time, 6),
        "Delta": delta if isinstance(delta, str) else round(delta, 6) if delta is not None else "不可用"
    })

# 输出结果
df = pd.DataFrame(results)
print("\nHeston 模型引擎对比结果：")
print(df)

# 计算 Black-Scholes 价格作为参考
flat_vol_ts = ql.BlackVolTermStructureHandle(
    ql.BlackConstantVol(today, calendar, 0.2, day_count)  # 假设波动率 sqrt(V0) = 0.2
)
bsm_process = ql.BlackScholesMertonProcess(spot_handle, dividend_ts, risk_free_ts, flat_vol_ts)
european_option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
bs_price = european_option.NPV()
print(f"\nBlack-Scholes 模型价格 (参考): {bs_price:.6f}")