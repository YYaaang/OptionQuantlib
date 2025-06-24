import QuantLib as ql
import numpy as np
from typing import Union

class QlCalendar:

    TIME_UNIT = ['days', 'weeks', 'months', 'quarters', 'years']

    def __init__(
            self,
            calendar='HKEx',
            init_date=ql.Date.todaysDate(),
            init_risk_free_rate=0.05,
            qlDayCounter:ql.DayCounter=ql.Actual365Fixed # ql.Business252  ql.Actual365Fixed
    ):
        self.callbacks_next_date = []
        # set market
        self.calendar = self.set_calendar(calendar)         # ql.Calendar
        # set dates
        self.init_date = self.set_today(init_date)
        self.all_trading_dates = np.array([self.init_date])
        self.day_counter = self.set_day_counter(self.calendar, qlDayCounter)     # ql.DayCounter
        # risk_free_rate and risk_free_rate_curve_handle
        self.risk_free_rate = ql.SimpleQuote(init_risk_free_rate)
        self.risk_free_rate_curve_handle = self._set_risk_free_curve()
        #
        #
        return

    def today(self):
        return ql.Settings.instance().evaluationDate

    def set_today(self, today):
        today = self._set_evalution_date(today)
        ql.Settings.instance().evaluationDate = today
        for callback in self.callbacks_next_date:
            callback()
        return today

    def to_next_trading_date(self):
        ql.Settings.instance().evaluationDate = self.calendar.adjust(self.today() + 1)
        for callback in self.callbacks_next_date:
            callback()
        return self.today()

    def register_callback(self, callback):
        self.callbacks_next_date.append(callback)

    def get_trading_dates(self, start_date, end: Union[ql.Date, int]=None, includeFirst=False):
        if isinstance(end, ql.Date):
            end_date = self.calendar.adjust(end, ql.Preceding)
            count = self.day_counter.dayCount(start_date, end_date)
        elif isinstance(end, int):
            count = end
        else:
            raise 'wrong end type'

        current = start_date

        sequence = [current := self.calendar.adjust(current + 1) for _ in range(count)]

        if includeFirst:
            sequence = [start_date] + sequence

        return np.array(sequence)

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
            init_date = self.today()

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

    while market.today() <= end_date:
        print(market.today())
        market.to_next_trading_date()
    pass
