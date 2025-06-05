import QuantLib as ql
import numpy as np

class QlCalendar:

    TIME_UNIT = ['days', 'weeks', 'months', 'quarters', 'years']

    def __init__(
            self,
            calendar='HKEx',
            init_date=ql.Date.todaysDate(),
            init_risk_free_rate=0.05,
            qlDayCounter:ql.DayCounter=ql.Business252
    ):
        # set market
        self.calendar = self.set_calendar(calendar)         # ql.Calendar
        # set dates
        self.init_date = self._set_evalution_date(init_date)
        self.today = self.init_date
        self.set_today(self.init_date)
        self.day_counter = self.set_day_counter(self.calendar, qlDayCounter)     # ql.DayCounter
        # risk_free_rate and risk_free_rate_curve_handle
        self.risk_free_rate = ql.SimpleQuote(init_risk_free_rate)
        self.risk_free_rate_curve_handle = self._set_risk_free_curve()
        #
        return

    def to_next_trading_date(self):
        self.today = self.calendar.adjust(self.today + 1)
        # print(f'next trading date: {self.today}')
        ql.Settings.instance().evaluationDate = self.today
        return

    def set_today(self, today):
        self.today = self._set_evalution_date(today)
        ql.Settings.instance().evaluationDate = self.today
        print('today: ', self.today)
        # if today >= self.today:
        #     self.today = self._set_evalution_date(today)
        #     ql.Settings.instance().evaluationDate = self.today
        # else:
        #     raise ValueError('today must be larger than past today')

    def set_risk_free_rate(self, new_risk_free_rate):
        self.risk_free_rate.setValue(new_risk_free_rate)
        # print(f'set new risk free rate: {self.risk_free_rate}')

    def _set_risk_free_curve(
            self,
            settlementDays=0,
    ):
        #
        risk_free_handle = ql.QuoteHandle(self.risk_free_rate)

        flat_curve = ql.FlatForward(
            settlementDays,
            self.calendar,
            risk_free_handle,  # 传入QuoteHandle
            self.day_counter
        )
        #
        risk_free_rate_curve_handle = ql.YieldTermStructureHandle(flat_curve)

        return risk_free_rate_curve_handle

    def set_calendar(self, calendar=None) -> ql.Calendar:
        if calendar is None:
            calendar = ql.HongKong(ql.HongKong.HKEx)
        elif calendar == 'HKEx':
            calendar = ql.HongKong(ql.HongKong.HKEx)
        else:
            calendar = calendar
        return calendar

    def yearFractionToDate(self, referenceDate, t):
        return ql.yearFractionToDate(self.day_counter, referenceDate, t)

    def set_day_counter(self, calendar, qlDayCounter):
        if qlDayCounter == ql.Business252:
            dayCounter = ql.Business252(calendar)     # ql.DayCounter
        else:
            # Actual365Fixed Actual360
            dayCounter = qlDayCounter()

        # one_year = 1
        #
        # new_date = ql.yearFractionToDate(dayCounter, self.init_date, one_year)
        #
        # dayCounter.one_year_all_days = self.calendar.businessDaysBetween(self.init_date, new_date)
        #
        # a = self.calendar.businessDaysBetween(self.init_date, new_date)
        # b = dayCounter.dayCount(self.init_date, new_date)
        #
        # check_year = dayCounter.yearFraction(self.init_date, new_date)
        #
        # if b != 252:
        #     raise 'xxxxxxxxxxxxx'
        #
        # if check_year != one_year:
        #     raise 'QLMarket set_day_counter wrong'

        return dayCounter


    def cal_date_advance(
            self,
            init_date: ql.Date = None,
            times: int = 1,
            time_unit: str = 'years'
    ):
        if init_date is None:
            init_date = self.today

        if time_unit in self.TIME_UNIT:
            if time_unit == 'days':
                new_date = self.calendar.advance(init_date, ql.Period(times, ql.Days), ql.Following)
            elif time_unit == 'weeks':
                new_date = self.calendar.advance(init_date, ql.Period(times, ql.Weeks), ql.Following)
            elif time_unit == 'months':
                new_date = self.calendar.advance(init_date, ql.Period(times, ql.Months), ql.Following)
            elif time_unit == 'quarters':
                new_date = self.calendar.advance(init_date, ql.Period(times * 3, ql.Months), ql.Following)
            elif time_unit == 'years':
                new_date = self.calendar.advance(init_date, ql.Period(times, ql.Years), ql.Following)
            else:
                raise 'wrong time_unit'
        else:
            raise 'wrong time_unit'
        return new_date

    # def cal_total_days(
    #         self,
    #         total_time:int = 1,
    #         time_unit:str = 'years'
    # ):
    #     new_date = self.cal_date_advance(self.today, total_time, time_unit)
    #
    #     timesteps = self.calendar.businessDaysBetween(self.today, new_date, False, True)
    #     return timesteps

    def _get_ql_time_unit(self, time_unit):
        if isinstance(time_unit, str):
            if time_unit == 'year':
                time_unit = ql.Years
            elif time_unit == 'months':
                time_unit = ql.Months
            elif time_unit == 'weeks':
                time_unit = ql.Weeks
            elif (time_unit == 'days') or (time_unit == 'day'):
                time_unit = ql.Days
        return time_unit

    def _set_evalution_date(self, day: ql.Date):
        if not isinstance(day, ql.Date):
            day = ql.Date(day)
        day = self.calendar.adjust(day)
        # ql.Settings.instance().evaluationDate = today
        return day

if __name__ == '__main__':

    start_date = ql.Date(1,1,2023)

    market = QlCalendar(
        calendar='HKEx',
        init_date=start_date,
        init_risk_free_rate=0.05,
        qlDayCounter = ql.Business252
    )

    start_date = market.init_date

    end_date = ql.Date(1,2,2023)

    while market.today <= end_date:
        print(market.today)
        market.to_next_trading_date()
    pass
