#%%
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.sandbox.distributions.genpareto import shape

current_path = os.getcwd()  # 获取当前工作目录
print("当前路径:", current_path)
current_path = current_path.replace('/samples', '')
print(sys.path)  # 显示所有模块搜索路径
sys.path.append(current_path)  # 添加自定义路径
print(sys.path)  # 显示所有模块搜索路径

def plot_fig(data):
    #1. 创建画布（调整大小适应大量曲线）
    plt.figure(figsize=(15, 8), dpi=100)
    
    # 2. 绘制所有折线（优化显示效果）
    for i, arr in enumerate(data):
        plt.plot(arr, 
                 linewidth=0.9,  # 细线避免重叠
                 alpha=0.3,      # 半透明区分重叠曲线
                 # color='blue'  # 统一颜色
                 ) 
    
    # 3. 添加图表元素
    plt.title(f"{n} Curves Visualization", fontsize=14)
    plt.xlabel("Index", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.5)  # 辅助网格线
    
    # 4. 显示/保存
    plt.tight_layout()  # 自动调整间距
    # plt.savefig('massive_lines.png', bbox_inches='tight')  # 保存高清图片
    plt.show()
#%% md
# # 金融衍生品定价模型测试
# 
# ### 本文件测试QlStocks类中的Black - Scholes、Black - Scholes - Merton和Heston模型实现
# 
#%% md
# ## 测试案例
#%%
import QuantLib as ql
from src.QlCalendar import QlCalendar
from src.QlStocks import QlStocks
#%%
# 初始化日历
start_date = ql.Date(1, 1, 2023)
ql_calendar = QlCalendar(init_date=start_date)
# 创建股票实例
s = QlStocks(ql_calendar)
#%% md
# ### 测试Black-Scholes模型
#%%
print("测试Black-Scholes模型:")
# 单个股票
s.black_scholes(100.0, sigma=0.25, code="AAPL")
# 多个股票
s.black_scholes([120.0, 80.0], sigma=[0.3, 0.2], code=["GOOG", "MSFT"])
s.df
#%% md
# ### 测试Black-Scholes-Merton模型
#%%
print("\n测试Black-Scholes-Merton模型:")
# 单个股票
s.black_scholes_merton(105.0, sigma=0.22, dividend_rate=0.03, code="AAPL_BSM")
# 多个股票
s.black_scholes_merton([115.0, 85.0], sigma=0.3, dividend_rate=[0.02, 0.04], code=["GOOG_BSM", "MSFT_BSM"])
s.df
#%% md
# ### 测试Heston模型
#%%
print("\n测试Heston模型:")
# 单个股票
s.heston(95.0, v0=0.04, kappa=1.0, theta=0.06, rho=-0.3, sigma=0.4, dividend_rate=0.01, code="AAPL_Heston")
# 多个股票
s.heston([110.0, 90.0],
         v0=[0.05, 0.03],
         kappa=[1.2, 0.8],
         theta=[0.05, 0.07],
         rho=[-0.2, -0.4],
         sigma=[0.3, 0.5],
         dividend_rate=[0.015, 0.025],
         code=["GOOG_Heston", "MSFT_Heston"])

#%% md
# ### 查看结果
#%%
# s.df
# #%%
# steps = 252
# #%%
# aapl = s.df.iloc[0]
# aapl
# #%%
# s.create_stock_path_generator(1, paths=1)
# #%%
# one_process = aapl['processes']
# process_generator = s.create_stock_path_generator(date_param = steps, process = one_process)
# one_random_path1 = np.array(process_generator.next().value())
# one_random_path1[:5], one_random_path1[-5:]
# #%%
# # 1. 生成示例数据（n个）
# n = 500
# data = [np.array(process_generator.next().value()) for _ in range(n)]  # 替换为你的实际数据
# #%%
# np.array(data)[:,-1].mean()
# #%%
# plot_fig(data)
# #%%
# aapl_bsm = s.df.iloc[3]
# #%%
# one_process = aapl_bsm['processes']
# process_generator = s.create_stock_path_generator(steps, process = one_process)
# n = 500
# data = [np.array(process_generator.next().value()) for _ in range(n)]
# print(np.array(data)[:,-1].mean())
# plot_fig(data)
# #%%
# appl_heston = s.df.iloc[6]
# appl_heston
# #%%
# one_process = appl_heston['processes']
# process_generator = s.create_stock_path_generator(steps, process = one_process)
# #%%
# n = 500
# data = [np.array(process_generator.next().value()) for _ in range(n)]
# #%%
# paths = [i[0] for i in data]
# #%%
# len(paths), paths[0].shape
# #%%
# print(np.array(paths)[:,-1].mean())
# #%%
# plot_fig(paths)
# #%%
# len(paths), len(paths[0])
#%%
# 多个股票
new_df = s.heston([95.0] * 500, v0=0.04, kappa=1.0, theta=0.06, rho=-0.3, sigma=0.4, dividend_rate=0.01)
print()
#%%
