import QuantLib as ql
import numpy as np

# 1. 市场参数配置
calculation_date = ql.Date(26, 5, 2025)
ql.Settings.instance().evaluationDate = calculation_date
spot_price = 100.0
risk_free_rate = 0.05
volatility = 0.20
maturity_date = ql.Date(26, 5, 2026)

# 2. 构建动态股价过程
stock_quote = ql.SimpleQuote(spot_price)
spot_handle = ql.QuoteHandle(stock_quote)
risk_free_curve = ql.YieldTermStructureHandle(
    ql.FlatForward(calculation_date, risk_free_rate, ql.Actual365Fixed()))
volatility_curve = ql.BlackVolTermStructureHandle(
    ql.BlackConstantVol(calculation_date, ql.NullCalendar(), volatility, ql.Actual365Fixed()))
process = ql.BlackScholesProcess(spot_handle, risk_free_curve, volatility_curve)

# 3. 蒙特卡洛引擎配置
time_steps = 252  # 1年交易日数
time_grid = ql.TimeGrid(1.0, time_steps)
rng = ql.GaussianRandomSequenceGenerator(
    ql.UniformRandomSequenceGenerator(time_steps, ql.UniformRandomGenerator(42)))
path_generator = ql.GaussianPathGenerator(process, time_grid, rng, False)


# 4. 日期切换与路径生成函数
def generate_path_with_dates(new_date, new_price):
    """更新日期和股价后生成路径"""
    ql.Settings.instance().evaluationDate = new_date
    stock_quote.setValue(new_price)
    path = path_generator.next().value()
    dates = [new_date + ql.Period(int(t * 365), ql.Days) for t in time_grid]
    return list(zip(dates, path))


# 5. 模拟5个交易日的路径变化
calendar = ql.China()
for day in range(1, 6):
    current_date = calendar.advance(calculation_date, ql.Period(day, ql.Days))
    current_price = 100 + day * 1.5  # 模拟每日上涨1.5
    path_data = generate_path_with_dates(current_date, current_price)

    # 打印首尾价格
    print(f"Date: {current_date.ISO()}, Spot: {current_price:.2f}")
    print(f"  Path Start: {path_data[0][1]:.2f}")
    print(f"  Path End: {path_data[-1][1]:.2f}\n")