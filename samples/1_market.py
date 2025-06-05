#%%
import QuantLib as ql
from src.QlMarket import QlMarket
#%%
start_date = ql.Date(1,1,2023)
start_date
#%% md
# # 香港市场
#%%
market = QlMarket(
    calendar='HKEx',
    init_date=start_date,
    init_risk_free_rate=0.05,
    qlDayCounter = ql.Business252
)

#%%
# 实际的起始日期
real_start_date = market.init_date
print(market.init_date)
print(market.risk_free_rate.value())
#%%
market.set_risk_free_rate(0.056)
print(market.risk_free_rate.value())
#%%
# 查看5个月后的日期
new_start_date = market.cal_date_advance(times=5, time_unit='months')
print(real_start_date, new_start_date)
#%%
market.set_today(new_start_date)
print(market.today)
#%%
one_year_later_date = market.cal_date_advance(init_date=market.today, times=1, time_unit='years')
print(one_year_later_date)
#%%
ql_calerder = market.calendar
#%%
business_days = ql_calerder.businessDaysBetween(market.today, one_year_later_date)
print(business_days)
#%%
ql_day_counter = market.day_counter
#%%
day_counts = ql_day_counter.dayCount(market.today, one_year_later_date)
print(day_counts)
#%%
day_friction = ql_day_counter.yearFraction(market.today, one_year_later_date)
print(day_friction)
#%% md
# ## day_friction is not 1, this is because ql.Business252 use 252 trading days as 1 year.
#%%
