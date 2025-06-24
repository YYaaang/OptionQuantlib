import yfinance as yf

# 获取小米期权数据
xiaomi = yf.Ticker("1810.HK")
option_dates = xiaomi.options

if option_dates:
    print("可用到期日：", option_dates)
    opt = xiaomi.option_chain(option_dates[0])
    print("看涨期权：\n", opt.calls[['strike', 'lastPrice', 'volume', 'impliedVolatility']].head())
else:
    print("无可用期权数据")