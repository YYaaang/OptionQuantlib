import QuantLib as ql
import pandas as pd
from datetime import datetime

# 1. 转换数据格式（假设您的数据存储在df中）
df = pd.read_csv(r'../data/calls.csv', index_col=0)

# 2. 设置QuantLib环境
calculation_date = ql.Date(23, 5, 2025)  # 当前日期设为2025-05-23
ql.Settings.instance().evaluationDate = calculation_date

# 市场参数（需要补充）
spot_price = 339.340  # 当前TSLA股价（需从市场获取）
risk_free_rate = 0.05  # 无风险利率
dividend_rate = 0.0  # 股息率

# 3. 创建BS模型框架
spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
flat_ts = ql.YieldTermStructureHandle(
    ql.FlatForward(calculation_date, risk_free_rate, ql.Actual365Fixed()))
dividend_ts = ql.YieldTermStructureHandle(
    ql.FlatForward(calculation_date, dividend_rate, ql.Actual365Fixed()))

# 4. 对每个期权合约进行计算
results = []
for _, row in df.iterrows():
    # 解析到期日（示例：从contractSymbol提取）
    expiry_date = ql.Date(30, 5, 2025)  # TSLA250530C...表示2025-05-30到期

    # 设置波动率曲面
    flat_vol = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(calculation_date, ql.NullCalendar(),
                            row['impliedVolatility'], ql.Actual365Fixed()))

    # 创建BS过程
    process = ql.BlackScholesMertonProcess(spot_handle, dividend_ts, flat_ts, flat_vol)

    # 构建欧式看涨期权
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, row['strike'])
    exercise = ql.EuropeanExercise(expiry_date)
    option = ql.VanillaOption(payoff, exercise)

    # 定价引擎
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

    # 计算结果
    results.append({
        'contract': row['contractSymbol'],
        'strike': row['strike'],
        'market_price': row['lastPrice'],
        'model_price': option.NPV(),
        'implied_vol': row['impliedVolatility'],
        'delta': option.delta(),
        'gamma': option.gamma(),
        'vega': option.vega() / 100,  # 每1%波动率变化
        'theta': option.theta() / 365  # 每日时间衰减
    })

# 5. 输出结果
result_df = pd.DataFrame(results)
print(result_df)