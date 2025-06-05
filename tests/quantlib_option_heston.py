import QuantLib as ql

today = ql.Date(7, ql.March, 2024)
ql.Settings.instance().evaluationDate = today

option = ql.EuropeanOption(
    ql.PlainVanillaPayoff(ql.Option.Call, 100.0),
    ql.EuropeanExercise(ql.Date(7, ql.June, 2024)),
)

u = ql.SimpleQuote(100.0)
r = ql.SimpleQuote(0.01)
σ = ql.SimpleQuote(0.20)

riskFreeCurve = ql.FlatForward(
    0, ql.TARGET(), ql.QuoteHandle(r), ql.Actual360()
)
volatility = ql.BlackConstantVol(
    0, ql.TARGET(), ql.QuoteHandle(σ), ql.Actual360()
)

process = ql.BlackScholesProcess(
    ql.QuoteHandle(u),
    ql.YieldTermStructureHandle(riskFreeCurve),
    ql.BlackVolTermStructureHandle(volatility),
)

engine = ql.AnalyticEuropeanEngine(process)

option.setPricingEngine(engine)

print(option.NPV())
print(option.delta())
print(option.gamma())
print(option.vega())

u.setValue(105.0)
print(option.NPV())

r.setValue(0.02)
print(option.NPV())

σ.setValue(0.15)
print(option.NPV())

u.setValue(105.0)
r.setValue(0.01)
σ.setValue(0.20)
print(option.NPV())

ql.Settings.instance().evaluationDate = ql.Date(7, ql.April, 2024)

print(option.NPV())
