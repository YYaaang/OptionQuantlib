# %% md
# QuantLib 美式期权定价示例
# 使用蒙特卡洛方法（MCAmericanEngine）计算美式期权价格及希腊字母

# %%
import QuantLib as ql
import numpy as np

# %% md
## 1. 设置基本参数
# %%
# 设置评估日期（当前日期）
today = ql.Date(31, 5, 2025)
ql.Settings.instance().evaluationDate = today

# 期权参数
spot_price = 100.0
strike_price = 105.0
volatility = 0.20
risk_free_rate = 0.05
dividend_rate = 0.02
maturity_date = ql.Date(31, 12, 2025)  # 2025年12月31日到期
option_type = ql.Option.Put  # 看跌期权

# %% md
## 2. 构建市场数据曲线
# %%
# 创建报价和曲线
spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
rate_handle = ql.YieldTermStructureHandle(
    ql.FlatForward(today, risk_free_rate, ql.Actual365Fixed()))
dividend_handle = ql.YieldTermStructureHandle(
    ql.FlatForward(today, dividend_rate, ql.Actual365Fixed()))
vol_handle = ql.BlackVolTermStructureHandle(
    ql.BlackConstantVol(today, ql.NullCalendar(), volatility, ql.Actual365Fixed()))

# 构建Black-Scholes过程
process = ql.BlackScholesMertonProcess(
    spot_handle, dividend_handle, rate_handle, vol_handle)

# %% md
## 3. 创建美式期权对象
# %%
payoff = ql.PlainVanillaPayoff(option_type, strike_price)
exercise = ql.AmericanExercise(today, maturity_date)
american_option = ql.VanillaOption(payoff, exercise)

# %% md
## 4. 使用蒙特卡洛引擎定价
# %%
# 设置蒙特卡洛参数
time_steps = 200  # 时间步数
n_paths = 1000  # 路径数量
seed = 42  # 随机种子

# 创建MCAmericanEngine
mc_engine = ql.MCAmericanEngine(
    process,
    "pseudorandom",  # 随机数生成器类型
    timeSteps=time_steps,
    requiredSamples=n_paths,
    seed=seed
)

american_option.setPricingEngine(mc_engine)

# %% md
## 5. 计算结果
# %%
print("美式期权定价结果:")
print(f"NPV: {american_option.NPV():.4f}")
print(f"Delta: {american_option.delta():.4f}")
print(f"Gamma: {american_option.gamma():.4f}")
print(f"Theta: {american_option.theta():.4f}")

# 计算Vega（波动率敏感度）
vol_handle = ql.BlackVolTermStructureHandle(
    ql.BlackConstantVol(today, ql.NullCalendar(), volatility+0.01, ql.Actual365Fixed()))
new_process = ql.BlackScholesMertonProcess(
    spot_handle, dividend_handle, rate_handle, vol_handle)
mc_engine = ql.MCAmericanEngine(
    new_process,
    "pseudorandom",
    timeSteps=time_steps,
    requiredSamples=n_paths,
    seed=seed
)
american_option.setPricingEngine(mc_engine)
vega = american_option.NPV() - original_price
print(f"Vega (1% vol change): {vega:.4f}")

# %% md
## 6. 与二叉树方法对比
# %%
# 使用二叉树方法作为基准
binomial_engine = ql.BinomialVanillaEngine(process, "crr", 500)
american_option.setPricingEngine(binomial_engine)
print("\n二叉树方法结果 (500步):")
print(f"NPV: {american_option.NPV():.4f}")