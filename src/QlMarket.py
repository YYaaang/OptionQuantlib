import QuantLib as ql
import numpy as np
from src.QlCalendar import QlCalendar
from src.QlStocks import QlStocks

class QlMarket:

    TIME_UNIT = ['days', 'weeks', 'months', 'quarters', 'years']

    def __init__(
            self,
            calendar='HKEx',
            init_date=ql.Date.todaysDate(),
            init_risk_free_rate=0.05,
            qlDayCounter:ql.DayCounter=ql.Business252
    ):
        self.ql_calendar = QlCalendar(
            calendar=calendar,
            init_date=init_date,
            init_risk_free_rate=init_risk_free_rate,
            qlDayCounter=qlDayCounter
        )
        self.ql_stocks = QlStocks(ql_calendar=self.ql_calendar)
