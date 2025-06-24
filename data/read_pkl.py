import pandas as pd


# data = pd.DataFrame(columns=['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_CLOSE'])
data0 = pd.read_pickle(r'data_20230101_20230331.pkl')
data1 = pd.read_pickle(r'data_20230401_20230630.pkl')
data2 = pd.read_pickle(r'data_20230701_20230930.pkl')
data3 = pd.read_pickle(r'data_20231001_20231231.pkl')
data_all = pd.concat([data0, data1, data2, data3])

data_all = data_all[['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_CLOSE']]


save_data = data_all[data_all['S_INFO_WINDCODE'].isin(['1810.HK', '0700.HK', '0981.HK'])]

save_data.to_pickle('stock_prices.pkl')

print()