#%%
import QuantLib as ql
import pandas as pd
#%%
today = ql.Date().todaysDate()
riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.05, ql.Actual365Fixed()))
dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.01, ql.Actual365Fixed()))

initialValue = ql.QuoteHandle(ql.SimpleQuote(100))

v0 = 0.005
theta = 0.010
kappa = 0.600
sigma = 0.400
rho = -0.15

hestonProcess = ql.HestonProcess(riskFreeTS, dividendTS, initialValue, v0, kappa, theta, sigma, rho)
hestonModel = ql.HestonModel(hestonProcess)
#%%
today
#%%
hestonModel
#%%
