
#%% md
# # 处理数据，仅考虑两个月内的期权，删除不必要的列

#%%
import sys
import os
current_path = os.getcwd()  # 获取当前工作目录
print("当前路径:", current_path)

all_files = os.listdir(current_path)
all_file_names = [i for i in all_files if i[-3:] != '.py']
all_file_names = [i for i in all_file_names if i[-3:] == 'csv']
#%%
import pandas as pd
import numpy as np
import time
# import QuantLib as ql
# from src.utils import str_dates_to_ql_dates
#%%

for file_name in all_file_names:
    print(file_name)
    data = pd.read_csv(file_name)
    columns = data.columns
    columns = [s.replace('[', '').replace(']', '').replace(' ', '') for s in columns]
    data.columns = columns
    #
    date_columns = ['QUOTE_READTIME', 'QUOTE_DATE', 'EXPIRE_DATE']
    numeric_cols = columns.copy()
    [numeric_cols.remove(i) for i in date_columns]
    for i in numeric_cols:
        data[i] = pd.to_numeric(data[i], errors='coerce')
    data = data.drop(columns=['C_SIZE','P_SIZE'])
    # 仅考虑两个月内的期权
    data = data[data['DTE'] < 63.0]

    # [ql.Date(i // 86400 + 25569) for i in data['QUOTE_UNIXTIME']]

    data = data.drop(columns=['QUOTE_READTIME', 'QUOTE_TIME_HOURS', 'QUOTE_TIME_HOURS', 'DTE'])
    #
    data.sort_values(['QUOTE_DATE', 'EXPIRE_DATE', 'STRIKE'], inplace=True)

    len_of_date = list(set(data['QUOTE_DATE'].str.len().values))
    if len(len_of_date) > 1:
        print('len_of_date', len_of_date)
    if len_of_date[0] == 11:
        data['QUOTE_DATE'] = data['QUOTE_DATE'].str[1:]
        data['EXPIRE_DATE'] = data['EXPIRE_DATE'].str[1:]
    else:
        print('length: ', len_of_date)
    # 写入
    file_name = file_name.replace('.csv', '.feather')
    data.to_feather(f'../{file_name}', compression='zstd')  # ZSTD压缩效率接近Parquet
    # 读取
    t0 = time.time()
    df = pd.read_feather(f'../{file_name}')
    print(file_name, "- time spent: ", time.time() - t0)
    data = 0
    df = 0
#%%
print()