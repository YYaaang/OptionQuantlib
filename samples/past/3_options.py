import QuantLib as ql
from src.QlMarket import QlMarket
from src.QlStockPricer import QlStockPricer
from src.QlVanillaOption import QlVanillaOption

start_date = ql.Date(1, 1, 2023)

qlMarket = QlMarket(init_date=start_date)

start_date = qlMarket.init_date

stock_1 = QlStockPricer(QlMarket=qlMarket, init_stock_price=100.0)
stock_1.using_Black_Scholes_model(sigma=0.2)

payoff = ql.PlainVanillaPayoff(ql.Option.Call, 100.0)
exercise = ql.EuropeanExercise(ql.Date(7, ql.June, 2024))

option = QlVanillaOption(
    payoff,
    exercise,
    QlStockPricer = stock_1,
    engine = ql.AnalyticEuropeanEngine
)

stock_price = option.stock_price
today = option.today

option0 = QlVanillaOption.init_from_price_and_date(
    strike_price=100.0,
    end_date=ql.Date(7, ql.June, 2024),
    QlStockPricer=stock_1,
    engine = ql.AnalyticEuropeanEngine
)
option0.NPV()

end_date = qlMarket.cal_date_advance(start_date,1, 'months')

steps = qlMarket.day_counter.dayCount(qlMarket.today, end_date)

stock_1.using_Heston_model(v0=0.005, kappa=0.8, theta=0.008, rho=0.2, sigma=0.1)


option2 = QlVanillaOption(
    payoff,
    exercise,
    stock_1,
    ql.MCEuropeanEngine,
    traits = "pseudorandom", # could use "lowdiscrepancy"
    timeSteps = steps,
    requiredSamples = 10000,
)

a = option2.NPV()
b = option2.delta()


option3 = QlVanillaOption(
    payoff,
    exercise,
    QlStockPricer = stock_1,
    engine = ql.MCEuropeanHestonEngine,
    traits = "pseudorandom", # could use "lowdiscrepancy"
    timeSteps = steps,
    requiredSamples = 10000,
)