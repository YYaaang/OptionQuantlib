import QuantLib as ql

calendar = ql.HongKong(ql.HongKong.HKEx)

start_date = ql.Date(3,1,2023)

end_date = ql.Date(3,1,2024)

dc = ql.Business252(calendar)

f1 = dc.yearFraction(start_date, end_date)

all_days1 = dc.dayCount(start_date, end_date)

f2 = all_days1 / 252

all_days2 = calendar.businessDaysBetween(start_date, end_date)

dc = ql.Business252()
d1 = ql.Date(1, 1, 2025)
d2 = ql.Date(31, 12, 2025)
print(dc.yearFraction(d1, d2))



# 设置日历和日期
calendar = ql.China(ql.China.SSE)
d1 = ql.Date(1, 1, 2024)
d2 = ql.Date(31, 12, 2024)

a = calendar.businessDaysBetween(d1, d2)
b = dc.dayCount(d1, d2)

# 选择DayCounter（需根据场景手动关联）
day_counter01 = ql.Business252(calendar)  # 交易日计数
print("交易日数量:", day_counter01.dayCount(d1, d2))

day_counter02 = ql.Actual360()  # 实际天数/实际天数
print(day_counter02.dayCount(d1, d2))  # 实际天数
print("年化分数:", day_counter02.yearFraction(d1, d2))
print()