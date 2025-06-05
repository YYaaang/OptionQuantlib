import QuantLib as ql
import numpy as np
import time

# 设置基础参数
num_prices = 10000
spot_prices = np.linspace(50.0, 150.0, num_prices)  # 1000 个价格
risk_free_rate = 0.05
dividend_rate = 0.02
volatility = 0.2
day_count = ql.Actual365Fixed()
calendar = ql.NullCalendar()
valuation_date = ql.Date(15, 6, 2023)
ql.Settings.instance().evaluationDate = valuation_date

# 创建共享的利率和波动率曲线
risk_free_curve = ql.FlatForward(valuation_date, risk_free_rate, day_count)
volatility_surface = ql.BlackConstantVol(valuation_date, calendar, volatility, day_count)
risk_free_handle = ql.YieldTermStructureHandle(risk_free_curve)
vol_handle = ql.BlackVolTermStructureHandle(volatility_surface)

# 计时：创建 BlackScholesProcess
start_time = time.time()
quotes = []
processes = []
for price in spot_prices:
    quote = ql.SimpleQuote(price)
    quote_handle = ql.QuoteHandle(quote)
    process = ql.BlackScholesProcess(quote_handle, risk_free_handle, vol_handle)
    quotes.append(quote)
    processes.append(process)
print(f"NumPy 方法创建 {num_prices} 个 BlackScholesProcess 用时: {time.time() - start_time:.4f} 秒")

# 计时：批量更改价格
new_spot_prices = np.linspace(60.0, 160.0, num_prices)  # 新价格
start_time = time.time()
for quote, new_price in zip(quotes, new_spot_prices):
    quote.setValue(new_price)
print(f"NumPy 方法批量更新 {num_prices} 个 SimpleQuote 用时: {time.time() - start_time:.4f} 秒")

# 示例：验证第一个进程的价格
print(f"验证第一个 BlackScholesProcess 的价格: {processes[0].x0()}")

import QuantLib as ql
import pandas as pd
import time

# 设置基础参数
spot_prices = pd.DataFrame({'price': np.linspace(50.0, 150.0, num_prices)})  # DataFrame 存储价格

# 计时：创建 BlackScholesProcess
start_time = time.time()
spot_prices['quote'] = spot_prices['price'].apply(lambda x: ql.SimpleQuote(x))
spot_prices['quote_handle'] = spot_prices['quote'].apply(lambda x: ql.QuoteHandle(x))
spot_prices['process'] = spot_prices['quote_handle'].apply(
    lambda x: ql.BlackScholesProcess(x, risk_free_handle, vol_handle)
)
print(f"DataFrame 方法创建 {num_prices} 个 BlackScholesProcess 用时: {time.time() - start_time:.4f} 秒")

# 计时：批量更改价格
new_spot_prices = np.linspace(60.0, 160.0, num_prices)
start_time = time.time()
spot_prices['price'] = new_spot_prices
for idx, row in spot_prices.iterrows():
    row['quote'].setValue(row['price'])
print(f"DataFrame 方法批量更新 {num_prices} 个 SimpleQuote 用时: {time.time() - start_time:.4f} 秒")


# 计时：批量更改价格
new_spot_prices = np.linspace(30.0, 200.0, num_prices)  # 新价格
start_time = time.time()
all_quotes = spot_prices['quote'].values
for quote, new_price in zip(all_quotes, new_spot_prices):
    quote.setValue(new_price)
print(f"DataFrame - NumPy 方法批量更新 {num_prices} 个 SimpleQuote 用时: {time.time() - start_time:.4f} 秒")


# 示例：验证第一个进程的价格
print(f"验证第一个 BlackScholesProcess 的价格: {spot_prices['process'].iloc[0].x0()}")
print()