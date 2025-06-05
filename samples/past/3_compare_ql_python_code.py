#%%

#%%
import numpy as np
import QuantLib as ql
from src.QlMarket import QlMarket
from src.QlStockPricer import QlStockPricer
from src.QlVanillaOption import QlVanillaOption
import matplotlib.pyplot as plt
import time

#%%
S0 = 100
strike = 100
r = 0.05
sigma = 0.3
T=1
#%%
start_date = ql.Date(1, 1, 2023)
qlMarket = QlMarket(init_date=start_date, init_risk_free_rate=r)
start_date = qlMarket.init_date
end_date = qlMarket.yearFractionToDate(start_date, 1)
start_date, end_date
#%%
qlMarket.calendar
#%%
stock_1 = QlStockPricer(QlMarket=qlMarket, init_stock_price=S0)
stock_1.using_Black_Scholes_model(sigma=sigma)

#%%
stock_1.day_counter.yearFraction(start_date, end_date)
#%% md
# ## compare option prices
#%% md
# ### Quantlib
#%%
ql_call = QlVanillaOption.init_from_price_and_date(
    strike_price=strike,
    end_date=end_date,
    QlStockPricer = stock_1,
    option_type=ql.Option.Call,
    engine = ql.AnalyticEuropeanEngine
)
ql_put = QlVanillaOption.init_from_price_and_date(
    strike_price=strike,
    end_date=end_date,
    QlStockPricer = stock_1,
    option_type=ql.Option.Put,
    engine = ql.AnalyticEuropeanEngine
)
print(f'call: {ql_call.NPV()}')
print(f'put: {ql_put.NPV()}')
print(f'call - put : {ql_call.NPV() - ql_put.NPV()}')
print(ql_call.NPV() - ql_put.NPV() - (S0 - strike * np.exp(-r * T)))
#%% md
# ### python code
#%%
from scipy.stats import norm

def bs_option(S0, strike, T, r, sigma, type):
    '''
    Black-Scholes option pricing formula

    Parameters
    ----------
    S0 : float
        Initial value of the underlying
    strike : float
        Strike price of the option
    T : float
        Time to maturity of the option
    r : float
        Risk-free interest rate
    sigma : float
        Volatility of the underlying
    type : str
        Type of option, either 'call' or 'put'
    '''
    d1 = (np.log(S0/strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if type == 'call':
        return S0 * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
    elif type == 'put':
        return strike * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError('Option type must be either "call" or "put"')

# test if fulfills put-call parity
call = bs_option(S0, strike, T, r, sigma, 'call')
put = bs_option(S0, strike, T, r, sigma, 'put')

print(f'call: {call}')
print(f'put: {put}')
print(f'call - put : {call - put}')
print(call - put - (S0 - strike * np.exp(-r * T)))
#%% md
# ## compare delta
#%% md
# ### Quantlib
#%%
call_delta = ql_call.delta()
put_delta = ql_put.delta()

print(call_delta)
print(put_delta)
print(call_delta - put_delta)
#%% md
# ### python code
#%%
def bs_delta(S0, strike, T, r, sigma, type):
    '''
    Black-Scholes delta

    Parameters
    ----------
    S0 : float
        Initial value of the underlying
    strike : float
        Strike price of the option
    T : float
        Time to maturity of the option
    r : float
        Risk-free interest rate
    sigma : float
        Volatility of the underlying
    type : str
        Type of option, either 'call' or 'put'
    '''
    d1 = (np.log(S0/strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    if type == 'call':
        return norm.cdf(d1)
    elif type == 'put':
        return norm.cdf(d1) - 1
    else:
        raise ValueError('Option type must be either "call" or "put"')

call_delta = bs_delta(S0, strike, T, r, sigma, 'call')
put_delta = bs_delta(S0, strike, T, r, sigma, 'put')

print(call_delta)
print(put_delta)
print(call_delta - put_delta)
#%% md
# ## stock path
#%%
steps = qlMarket.day_counter.dayCount(start_date, end_date)
dt =  1 / steps
M = 10000
dt
#%% md
# ### Quantlib
# ### 循环会慢些
#%%
# stock_path_generator1 = stock_1.create_stock_path_generator(steps)
# ql_stock_path = [np.array(stock_path_generator1.next().value()) for _ in range(M)]
# ql_stock_path = np.array(ql_stock_path)
#%%
# ql_stock_path = np.array(ql_stock_path)
# ql_stock_path = ql_stock_path.T
# ql_stock_path.shape
#%%
# ql_stock_path[-1].mean(), ql_stock_path[-1].var()
#%%
# plt.plot(ql_stock_path)
#%% md
# ### python code
#%%
def brownian(T, dt, M):
    ''' 
    Generate a brownian path with from 0 to T with time step dt
    
    Based on:
        X(t+dt) = X(t) + N(0, delta**2 * dt)

    Code adapted from:
    "https://scipy-cookbook.readthedocs.io/items/BrownianMotion.html"

    Parameters
    ----------
    T : float
        Final time of the brownian path
    dt : float
        Time step of the brownian path
    M : int
        Number of paths to generate
    '''

    K = int(T/dt)
    X = np.zeros((M, K+1))

    # Create the increments of the brownian motion
    r = norm.rvs(size=(M, K), scale=np.sqrt(dt))
    
    # Cumulative sum of the random numbers
    X[:, 1:] = np.cumsum(r, axis=1)

    return X.T

def geometric_brownian_motion(x0, T, dt, sigma, mu, M):
    brownian_path = brownian(T, dt, M)
    return x0 * np.exp((mu - 0.5 * sigma**2) * dt + sigma * brownian_path)
#%%
stock_path = geometric_brownian_motion(S0, T, dt, sigma, r, M)
#%%
stock_path.shape
#%%
stock_path[-1].mean(), stock_path[-1].var()
#%%
# plt.plot(stock_path)
#%% md
# ## Compare options and pnl
# ### we use stock_price
#%% md
# ## QuantLib
#%%

#%%
# Generate the initial option price
ql_option_price = ql_call.NPV()

# Generate the initial delta
initial_delta = ql_call.delta()

# Generate the cash account
cash = np.zeros(stock_path.shape)

# Generate the portfolio value
portfolio = np.zeros(stock_path.shape)

# PnL
pnl = np.zeros(stock_path.shape)

# initial cash is the option price * 100 - delta * S0
cash[0] = ql_option_price * steps - initial_delta * S0

# initial portfolio value is the - option price * 100 + delta * S0
portfolio[0] = - ql_option_price * steps + initial_delta * S0

option_path = np.zeros(stock_path.shape)
option_path[0] = ql_option_price
option_delta = np.zeros(stock_path.shape)
option_delta[0] = initial_delta

# 第二天
qlMarket.to_next_trading_date()
t = 1

def my_func(x):
    ql_call.stock_price.setValue(x)
    return ql_call.NPV()

vectorized_func = np.vectorize(my_func)

while qlMarket.today <= end_date:
    # print(qlMarket.today)
    the_price = stock_path[t]
    t = time.time()
    new_price0 = vectorized_func(the_price)
    print('1', time.time() - t)
    t = time.time()
    new_price = np.zeros(the_price.shape)
    for i, v in enumerate(the_price):
        ql_call.stock_price.setValue(v)
        new_price[i] = ql_call.NPV()
    print('2', time.time() - t)
    t = time.time()
    option_price = bs_option(the_price, strike, T - t * dt, r, sigma, 'call')
    print('3', time.time() - t)

    all_stocks = []
    for i in the_price:
        sp = QlStockPricer(QlMarket=qlMarket, init_stock_price=i)
        sp.using_Black_Scholes_model(sigma=sigma)
        all_stocks.append(sp)
    all_options = [
        QlVanillaOption.init_from_price_and_date(
            strike_price=strike,
            end_date=end_date,
            QlStockPricer=s,
            option_type=ql.Option.Call,
            engine=ql.AnalyticEuropeanEngine
        )
        for s in all_stocks
    ]
    all_options_list = []
    for i in range(len(all_stocks)):
        option = QlVanillaOption.init_from_price_and_date(
            strike_price=strike,
            end_date=end_date,
            QlStockPricer=all_stocks[i],
            option_type=ql.Option.Call,
            engine=ql.AnalyticEuropeanEngine
        )
        all_options_list.append(option)

    [s.stock_price.setValue(the_price[i]) for i, s in enumerate(all_stocks)]

    a = [s.stock_price.value() for i, s in enumerate(all_stocks)]

    option_prices = [option.NPV() for option in all_options]

    for i, v in enumerate(new_price0):
        if new_price[i] != v:
            print()
    print()


    t += 1
    qlMarket.to_next_trading_date()

for t in range(1, len(stock_path)):
    # Update the stock price
    stock_price = stock_path[t]

    # Update the option price
    option_price = bs_option(stock_price, strike, T - t * dt, r, sigma, 'call')
    option_path[t] = option_price

    # Update the delta
    delta = bs_delta(stock_price, strike, T - t * dt, r, sigma, 'call')
    option_delta[t] = delta

    # Update the portfolio value
    portfolio[t] = - option_price * M + initial_delta * stock_price

    # Update the cash account
    cash[t] = cash[t-1] * np.exp(r * dt)

    # Update the PnL
    pnl[t] = portfolio[t] + cash[t]

# Plot the PnL, cash account and portfolio value
# Plot in separate figure the stock price and option price
fig, ax = plt.subplots(2, 3, figsize=(18, 8))

ax[0, 0].plot(stock_path)
ax[0, 0].set_title('Stock price')

ax[0, 1].plot(option_path)
ax[0, 1].set_title('Option price')

ax[0, 2].plot(option_delta)
ax[0, 2].set_title('Option delta')

ax[1, 0].plot(portfolio)
ax[1, 0].set_title('Portfolio value')

ax[1, 1].plot(cash)
ax[1, 1].set_title('Cash account')

ax[1, 2].plot(pnl)
ax[1, 2].set_title('PnL')

plt.show()


# check distribution of final PnL
plt.hist(pnl[-1], bins=50)
plt.title('Distribution of final PnL')
plt.show()
#%% md
# ## python code
#%%
# Generate the initial option price
option_price = bs_option(S0, strike, T, r, sigma, 'call')

# Generate the initial delta
initial_delta = bs_delta(S0, strike, T, r, sigma, 'call')

# Generate the cash account
cash = np.zeros(stock_path.shape)

# Generate the portfolio value
portfolio = np.zeros(stock_path.shape)

# PnL
pnl = np.zeros(stock_path.shape)

# initial cash is the option price * 100 - delta * S0
cash[0] = option_price * steps - initial_delta * S0

# initial portfolio value is the - option price * 100 + delta * S0
portfolio[0] = - option_price * steps + initial_delta * S0

option_path = np.zeros(stock_path.shape)
option_path[0] = option_price
option_delta = np.zeros(stock_path.shape)
option_delta[0] = initial_delta

for t in range(1, len(stock_path)):
    # Update the stock price
    stock_price = stock_path[t]

    # Update the option price
    option_price = bs_option(stock_price, strike, T - t * dt, r, sigma, 'call')
    option_path[t] = option_price

    # Update the delta
    delta = bs_delta(stock_price, strike, T - t * dt, r, sigma, 'call')
    option_delta[t] = delta

    # Update the portfolio value
    portfolio[t] = - option_price * M + initial_delta * stock_price

    # Update the cash account
    cash[t] = cash[t-1] * np.exp(r * dt)

    # Update the PnL
    pnl[t] = portfolio[t] + cash[t]

# Plot the PnL, cash account and portfolio value
# Plot in separate figure the stock price and option price
fig, ax = plt.subplots(2, 3, figsize=(18, 8))

ax[0, 0].plot(stock_path)
ax[0, 0].set_title('Stock price')

ax[0, 1].plot(option_path)
ax[0, 1].set_title('Option price')

ax[0, 2].plot(option_delta)
ax[0, 2].set_title('Option delta')

ax[1, 0].plot(portfolio)
ax[1, 0].set_title('Portfolio value')

ax[1, 1].plot(cash)
ax[1, 1].set_title('Cash account')

ax[1, 2].plot(pnl)
ax[1, 2].set_title('PnL')

plt.show()


# check distribution of final PnL
plt.hist(pnl[-1], bins=50)
plt.title('Distribution of final PnL')
plt.show()
#%%
stock_path.shape
#%%
