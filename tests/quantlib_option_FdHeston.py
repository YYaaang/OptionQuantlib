import QuantLib as ql

# 1. 设置基础参数
today = ql.Date().todaysDate()
ql.Settings.instance().evaluationDate = today

# 2. 构建Heston模型（用户已提供部分）
riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.05, ql.Actual365Fixed()))
dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.01, ql.Actual365Fixed()))
stock_price = ql.SimpleQuote(100)
initialValue = ql.QuoteHandle(stock_price)
v0, kappa, theta, rho, sigma = 0.005, 0.8, 0.008, 0.2, 0.1

hestonProcess = ql.HestonProcess(riskFreeTS, dividendTS, initialValue, v0, kappa, theta, sigma, rho)
hestonModel = ql.HestonModel(hestonProcess)

# 3. 配置有限差分引擎（用户已提供）
tGrid, xGrid, vGrid = 100, 100, 50
dampingSteps = 0
fdScheme = ql.FdmSchemeDesc.ModifiedCraigSneyd()
engine = ql.FdHestonVanillaEngine(hestonModel, tGrid, xGrid, vGrid, dampingSteps, fdScheme)

# 4. 创建欧式看涨期权
strike_price = 100
maturity_date = today + ql.Period(1, ql.Years)  # 1年后到期

payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike_price)
exercise = ql.EuropeanExercise(maturity_date)
option = ql.VanillaOption(payoff, exercise)

# 5. 设置定价引擎并计算Delta
option.setPricingEngine(engine)

print(f"期权价格: {option.NPV():.4f}")
print(f"Delta: {option.delta():.4f}")  # FdHestonVanillaEngine直接支持Delta计算
print(f"Gamma: {option.gamma():.4f}")  # 其他希腊字母也可直接获取

def finite_difference_delta(option, spot_quote, dS=0.01):
    S0 = spot_quote.value()
    spot_quote.setValue(S0 + dS)
    P_up = option.NPV()
    spot_quote.setValue(S0 - dS)
    P_down = option.NPV()
    spot_quote.setValue(S0)  # 恢复原值
    return (P_up - P_down) / (2 * dS)

manual_delta = finite_difference_delta(option, stock_price)
print(f"手动差分Delta: {manual_delta:.4f}")

print()