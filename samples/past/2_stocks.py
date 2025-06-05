# %% md
# # 金融衍生品定价模型测试

# 本文件测试QlStocks类中的Black - Scholes、Black - Scholes - Merton和Heston模型实现

# %% md
# ## 测试案例
# %%
import QuantLib as ql
from src.QlCalendar import QlCalendar
from src.QlStocks import QlStocks
# %%
# 初始化日历
start_date = ql.Date(1, 1, 2023)
ql_calendar = QlCalendar(init_date=start_date)

# 创建股票实例
s = QlStocks(ql_calendar)

# %% md
# ### 测试Black-Scholes模型
# %%
print("测试Black-Scholes模型:")
# 单个股票
s.black_scholes(100.0, sigma=0.25, code="AAPL")
# 多个股票
s.black_scholes([120.0, 80.0], sigma=[0.3, 0.2], code=["GOOG", "MSFT"])

# %% md
# ### 测试Black-Scholes-Merton模型
# %%
print("\n测试Black-Scholes-Merton模型:")
# 单个股票
s.black_scholes_merton(105.0, sigma=0.22, dividend_rate=0.03, code="AAPL_BSM")
# 多个股票
s.black_scholes_merton([115.0, 85.0], sigma=0.3, dividend_rate=[0.02, 0.04], code=["GOOG_BSM", "MSFT_BSM"])

# %% md
# ### 测试Heston模型
# %%
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

# %% md
# ### 查看结果
# %%
print("\n最终结果:")
print(s.df)