

#%%
import numpy as np
import QuantLib as ql
from src.QlMarket import QlMarket
from src.QlStockPricer import QlStockPricer
from src.QlVanillaOption import QlVanillaOption

#%%
S0 = 100
strike = 100
r = 0.03
T=1
#%%
start_date = ql.Date(1, 1, 2023)
qlMarket = QlMarket(init_date=start_date, init_risk_free_rate=r)
end_date = qlMarket.cal_date_advance(times=T, time_unit='years')
#%%
stock_1 = QlStockPricer(QlMarket=qlMarket, init_stock_price=S0)
stock_1.using_Black_Scholes_model(sigma=0.2)

#%%
option_call = QlVanillaOption.init_from_price_and_date(
    strike_price=strike,
    end_date=end_date,
    QlStockPricer = stock_1,
    option_type=ql.Option.Call,
    engine = ql.AnalyticEuropeanEngine
)
option_put = QlVanillaOption.init_from_price_and_date(
    strike_price=strike,
    end_date=end_date,
    QlStockPricer = stock_1,
    option_type=ql.Option.Put,
    engine = ql.AnalyticEuropeanEngine
)
#%%
# # test if fulfills put-call parity

#%%
print(option_call.NPV() - option_put.NPV() - (S0 - strike * np.exp(-r * T)))



#%%
print()
