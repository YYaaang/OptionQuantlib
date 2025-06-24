import pandas as pd
import numpy as np
import QuantLib as ql
import time
from config.config import RANDOM_SEED
from typing import Union, List

from src.utils import check_list_length, set_option_name
from src.QlCalendar import QlCalendar
from src.QlEuropeanOption import QlEuropeanOption

class QlStock:
    def __init__(
            self,
            ql_calendar: QlCalendar,
            code,
            price_quote: ql.SimpleQuote,
            dividend_quote,
            volatility,
            process_type,
            process
    ):
        self.ql_calendar = ql_calendar
        self.code = code
        self.price_quote = price_quote
        self.dividend_quote = dividend_quote
        self.volatility:ql.SimpleQuote = volatility
        self.process_type = process_type
        self.process = process

        self.stock_prices = pd.DataFrame()

        return

    def get_price_quote(self):
        return self.price_quote

    def get_price(self):
        return self.price_quote.value()

    def set_today_price(
            self,
            price:float,
    ):
        self.price_quote.setValue(price)
        return

    def set_volatility(self, volatility:float):
        self.volatility.setValue(volatility)
        return

    def set_european_option(
            self,
            option_type: Union[str] = None,
            strike_price: Union[float] = None,
            maturity_date: Union[ql.Date] = None,
            code=None,
            qlExercise: ql.Exercise = ql.EuropeanExercise,
            engine_type: str = None,  # 'Analytic', 'Fd', 'MC'
    ):
        if option_type == 'call':
            option_type = ql.Option.Call
        elif option_type == 'put':
            option_type = ql.Option.Put

        payoff = ql.PlainVanillaPayoff(option_type, strike_price)
        exercise = qlExercise(maturity_date)

        option = ql.EuropeanOption(payoff, exercise)

        code = set_option_name(self.code, maturity_date, strike_price,
                               option_type
                               )

        op = QlEuropeanOption(
            code = code,
            option_type = option_type,
            strike = strike_price,
            maturity = maturity_date,
            option = option,
            stock_price_quote = self.price_quote,
            process_type = self.process_type,
            process = self.process,
        )

        op.stock_vol_quote = self.volatility

        if engine_type is not None:
            if engine_type == 'Analytic':
                op.analytic_engine()
            elif engine_type == 'Fd':
                op.fd_engine()
        return op

    def stock_paths(
            self,
            paths,
            date_param: [int, ql.Date],  # timesteps, end_date
            random_seed=RANDOM_SEED
    ):
        stock_path_generator = self.stock_path_generator(
            date_param = date_param,
            random_seed = random_seed
        )
        n = self.process.factors()
        if n == 1:
            stock_paths = np.array([np.array(stock_path_generator.next().value()) for _ in np.arange(paths)])
        else:
            stock_path_generator.next().value()
            stock_paths = np.array([np.array(stock_path_generator.next().value()[0]) for _ in np.arange(paths)])
        return stock_paths

    def stock_path_generator(
            self,
            date_param:[int, ql.Date], #  timesteps, end_date
            random_seed=RANDOM_SEED
    ):
        timesteps, total_time = self._set_date_parm(date_param)

        # 创建路径生成器
        stock_path_generator = self._set_path_generator(total_time, timesteps, random_seed=random_seed)
        # 生成1条路径
        # stock_path = stock_path_generator.next().value()
        return stock_path_generator

    def _set_path_generator(self, length, timesteps, random_seed=RANDOM_SEED):
        #
        pathGenerator = None
        if type(self.process) in [ql.BlackScholesMertonProcess, ql.BlackScholesProcess]:
            urng = ql.UniformRandomGenerator(random_seed)
            sequenceGenerator = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(timesteps, urng))
            pathGenerator = ql.GaussianPathGenerator(
                self.process, length, timesteps, sequenceGenerator, False)
        elif type(self.process) is ql.HestonProcess:
            dimension = self.process.factors()
            rng = ql.UniformLowDiscrepancySequenceGenerator(dimension * timesteps, random_seed)
            sequenceGenerator = ql.GaussianLowDiscrepancySequenceGenerator(rng)
            time_grid = ql.TimeGrid(length, timesteps)
            pathGenerator = ql.GaussianSobolMultiPathGenerator(
                self.process, time_grid, sequenceGenerator, False)

        # np.array(pathGenerator.next().value())
        return pathGenerator

    def _set_date_parm(self, date_param):

        if isinstance(date_param, int):
            timesteps = date_param
            end_date = self.ql_calendar.cal_date_advance(init_date=self.ql_calendar.today(), times=timesteps, time_unit='days')
            print(f"使用步数: {timesteps} 步")
        elif isinstance(date_param, ql.Date):
            end_date = self.ql_calendar._set_evalution_date(date_param)
            timesteps = self.ql_calendar.day_counter.dayCount(self.ql_calendar.today(), end_date)
            print(f"使用结束日期: {end_date}")
        else:
            raise TypeError("参数必须是 float（total_length）或 ql.Date（end_date）")

        total_time = self.ql_calendar.day_counter.yearFraction(self.ql_calendar.today(), end_date)

        print(f'timesteps: {timesteps}, Time length(per year): {total_time}'
              f'start_date: {self.ql_calendar.today()} end_date: {end_date}')

        return timesteps, total_time