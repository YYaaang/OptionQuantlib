
import QuantLib as ql
import numpy as np
from src.QlMarket import QlMarket
from config.config import RANDOM_SEED

class QlStockPricer:

    def __init__(
            self,
            QlMarket:QlMarket,
            init_stock_price:[]=100.0,  # Initial stock price
            trading_time_unit = '1d',
    ):
        # set market
        self.QlMarket = QlMarket
        # set price
        self.init_stock_price = init_stock_price
        self.stock_price = ql.SimpleQuote(init_stock_price)

        self.today = self.QlMarket.today
        self.trading_time_unit = trading_time_unit
        self.calendar = self.QlMarket.calendar  # ql.Calendar
        self.day_counter = self.QlMarket.day_counter     # ql.DayCounter
        #
        self.risk_free_rate_curve_handle = self.QlMarket.risk_free_rate_curve_handle

        #
        self.model_type = ''
        print('Next step, you need set model type')
        self.process = None
        self.dividend_rate = ql.SimpleQuote(0)
        self.dividend_curve_handle = self._set_dividend_curve()
        self.sigma = None

        #
        self.schedule = None

        return

    def using_Black_Scholes_model(self, sigma=0.2):
        if self.model_type == 'Black_Scholes':
            self.sigma.setValue(sigma)
            print(f'Black Scholes model set sigma: {sigma}')
            return

        self.model_type = 'Black_Scholes'
        self.sigma = ql.SimpleQuote(sigma)

        volatility = ql.BlackConstantVol(0, self.calendar, sigma, self.day_counter)
        volatility_handle = ql.BlackVolTermStructureHandle(volatility)

        self.process = ql.BlackScholesProcess(
            ql.QuoteHandle(self.stock_price),
            self.risk_free_rate_curve_handle,
            volatility_handle
        )
        print(f'set new model: Black Scholes model with sigma: {sigma}')
        return

    def using_Black_Scholes_Merton_model(self, sigma=0.2, dividend_rate=None):
        if dividend_rate is not None:
            self.dividend_rate.setValue(dividend_rate)

        if self.model_type == 'Black_Scholes_Merton':
            self.sigma.setValue(sigma)
            print(f'Black Scholes Merton model set yield rate: {dividend_rate}, new sigma: {sigma}')
            return

        self.model_type = 'Black_Scholes_Merton'
        self.sigma = ql.SimpleQuote(sigma)

        volatility = ql.BlackConstantVol(0, self.calendar, sigma, self.day_counter)
        volatility_handle = ql.BlackVolTermStructureHandle(volatility)

        self.process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(self.stock_price),
            self.dividend_curve_handle,
            self.risk_free_rate_curve_handle,
            volatility_handle
        )
        print(f'set new model: Black Scholes Merton model with sigma: {sigma}, yield rate: {dividend_rate}')
        return

    def using_Heston_model(self, v0=0.005, kappa=0.8, theta=0.008, rho=0.2, sigma=0.1, dividend_rate=None):
        if dividend_rate is not None:
            self.dividend_rate.setValue(dividend_rate)

        self.model_type = 'Heston'
        self.sigma = None

        self.process = ql.HestonProcess(
            self.risk_free_rate_curve_handle,
            self.dividend_curve_handle,
            ql.QuoteHandle(self.stock_price),
            v0,
            kappa,
            theta,
            sigma,
            rho
        )

        print(f'set new model: Heston model with yield rate: v0: {v0}, kappa: {kappa}, theta: {theta}, rho: {rho}, sigma: {sigma}, {dividend_rate}')
        return

    def create_stock_path_generator(
            self,
            date_param:[int, ql.Date], #  timesteps, end_date
    ):
        if isinstance(date_param, int):
            timesteps = date_param
            end_date = self.QlMarket.cal_date_advance(init_date=self.today, times=timesteps, time_unit='days')
            print(f"使用步数: {timesteps} 步")
        elif isinstance(date_param, ql.Date):
            end_date = self.QlMarket._set_evalution_date(date_param)
            timesteps = self.day_counter.dayCount(self.today, end_date)
            print(f"使用结束日期: {end_date}")
        else:
            raise TypeError("参数必须是 float（total_length）或 ql.Date（end_date）")

        total_time = self.day_counter.yearFraction(self.today, end_date)

        print(f'timesteps: {timesteps}, Time length(per year): {total_time}'
              f'start_date: {self.today} end_date: {end_date}')

        # 创建路径生成器
        stock_path_generator = self._set_path_generator(total_time, timesteps, self.process)
        # 生成1条路径
        stock_path = stock_path_generator.next().value()
        return stock_path_generator

    def get_process(self):
        return self.process

    def _set_dividend_curve(
            self,
            settlementDays=0,
    ):
        dividend_curve_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(
                settlementDays,
                self.calendar,
                ql.QuoteHandle(self.dividend_rate),
                self.day_counter
            )
        )
        return dividend_curve_handle


    def _set_volatility(self):
        volatility = ql.BlackConstantVol(0, self.calendar, self.sigma, self.day_counter)
        volatility_handle = ql.BlackVolTermStructureHandle(volatility)
        return volatility_handle

    def _set_process(self, ql_stock_price, risk_free_rate_curve_handle, volatility_curve_handle):
        #
        process = ql.BlackScholesProcess(
            ql.QuoteHandle(ql_stock_price),
            risk_free_rate_curve_handle,
            volatility_curve_handle
        )
        return process

    def _set_path_generator(self, length, timesteps, process):
        #
        pathGenerator = None
        if type(process) in [ql.BlackScholesMertonProcess, ql.BlackScholesProcess]:
            urng = ql.UniformRandomGenerator(RANDOM_SEED)
            sequenceGenerator = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(timesteps, urng))
            pathGenerator = ql.GaussianPathGenerator(
                process, length, timesteps, sequenceGenerator, False)
        elif type(process) is ql.HestonProcess:
            dimension = process.factors()
            rng = ql.UniformLowDiscrepancySequenceGenerator(dimension * timesteps, RANDOM_SEED)
            sequenceGenerator = ql.GaussianLowDiscrepancySequenceGenerator(rng)
            time_grid = ql.TimeGrid(length, timesteps)
            pathGenerator = ql.GaussianSobolMultiPathGenerator(
                process, time_grid, sequenceGenerator, False)

        # np.array(pathGenerator.next().value())
        return pathGenerator

    @classmethod
    # Black-Scholes公式计算期权价格
    def BSM(cls, S0, K, T, r, sigma):
        from math import log, sqrt, exp
        from scipy.stats import norm
        d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / sigma / sqrt(T)
        d2 = d1 - sigma * sqrt(T)
        Callprice = S0 * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        print("BS公式计算看涨期权价格: %.4f" % Callprice)
        return Callprice



