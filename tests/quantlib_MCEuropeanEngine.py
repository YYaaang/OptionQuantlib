import QuantLib as ql

today = ql.Date().todaysDate()
riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.05, ql.Actual365Fixed()))
dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.01, ql.Actual365Fixed()))
volatility = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), 0.1, ql.Actual365Fixed()))
initialValue = ql.QuoteHandle(ql.SimpleQuote(100))
process = ql.BlackScholesMertonProcess(initialValue, dividendTS, riskFreeTS, volatility)

steps = 2
rng = "pseudorandom" # could use "lowdiscrepancy"
numPaths = 1000

engine = ql.MCEuropeanEngine(process, rng, steps, requiredSamples=numPaths, enableGreeks=True)

#
strike = 100.0
maturity = ql.Date(15,6,2025)
option_type = ql.Option.Call

payoff = ql.PlainVanillaPayoff(option_type, strike)
binaryPayoff = ql.CashOrNothingPayoff(option_type, strike, 1)

europeanExercise = ql.EuropeanExercise(maturity)
europeanOption = ql.VanillaOption(payoff, europeanExercise)
europeanOption.setPricingEngine(engine)

a = europeanOption.NPV()
b = europeanOption.delta()
print()
