import QuantLib as ql
import numpy as np


class OptionPricer:

    def __init__(
            self,
            init_stock_price=100.0,  # Initial stock price
            strike_price=100.0,  # Option strike price
            yield_rate=0,  #
            risk_free_rate=0.05,  # Risk-free rate
            sigma=0.20,  # Volatility
            maturity_time=1,  # Option maturity (years)
            maturity_time_unit='year',
            trading_time_unit = '1d',
            calendar='HKEx',
            # trading_days=252,  # Trading days per year
            init_date=ql.Date.todaysDate()
    ):
        self.init_stock_price = init_stock_price
        self.ql_stock_price = ql.SimpleQuote(init_stock_price)

        self.strike_price = strike_price
        self.yield_rate = yield_rate
        self.risk_free_rate = risk_free_rate
        self.sigma = sigma
        self.maturity_time = maturity_time
        self.maturity_time_unit = maturity_time_unit
        self.trading_time_unit = trading_time_unit
        self.calendar = self.set_calendar(calendar)         # ql.Calendar
        self.dayCounter = ql.Business252(self.calendar)     # ql.DayCounter
        # self.trading_days = trading_days
        # set dates
        self.init_date = self._set_evalution_date(init_date)
        self.today = self.init_date
        maturity_time_unit = self._get_ql_time_unit(self.maturity_time_unit)
        self.maturity_date = self.today + ql.Period(int(self.maturity_time), maturity_time_unit)
        self.total_trading_days = self.calendar.businessDaysBetween(self.init_date, self.maturity_date)
        #
        self.risk_free_curve = self._set_riskFreeCurve()
        self.volatility = self._set_volatility()
        self.process = self._set_process(self.ql_stock_price, self.risk_free_curve, self.volatility)
        self.schedule = None
        return

    def set_calendar(self, calendar=None) -> ql.Calendar:
        if calendar is None:
            calendar = ql.HongKong(ql.HongKong.HKEx)
        elif calendar == 'HKEx':
            calendar = ql.HongKong(ql.HongKong.HKEx)
        else:
            calendar = calendar
        return calendar

    # def set_expiration_time_in_years(self, T):
    #     self.T = T

    # def set_steps(self, steps):

    def create_stock_path_generator(
            self,
    ):
        # 创建随机数生成器
        rng = 42  # 设定种子保证可重复性
        # 创建路径生成器
        stock_path_generator = self._set_path_generator(self.total_trading_days, self.process, rng)
        # 生成1条路径
        stock_path = stock_path_generator.next().value()

        return stock_path_generator


    def calculate_stock_path(
            self,

    ):

        return

    def calculate_option(
            self,
            stock_price,
            new_date,
            option_type='call'
    ):
        self.ql_stock_price.setValue(stock_price)
        self._set_evalution_date(new_date)
        # 构建期权
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, self.strike_price)
        exercise = ql.EuropeanExercise(self.maturity_date)
        option = ql.EuropeanOption(payoff, exercise)

        # engine
        engine = ql.AnalyticEuropeanEngine(self.process)
        option.setPricingEngine(engine)
        c = option.NPV()
        # 计算希腊字母
        print('-----------------------------------------')
        print(f"stock_price: {stock_price}, new_date: {new_date}, option_type: {option_type}, 初始定价: {option.NPV():.4f}")
        print("Grek:")
        print("%-12s: %4.4f" % ("Delta", option.delta()))
        print("%-12s: %4.4f" % ("Gamma", option.gamma()))
        print("%-12s: %4.4f" % ("Vega", option.vega()))
        print('-----------------------------------------')
        return option

    def _set_riskFreeCurve(self):
        risk_free_curve = ql.FlatForward(0, self.calendar, self.risk_free_rate, self.dayCounter)
        return risk_free_curve

    def _set_volatility(self):
        volatility = ql.BlackConstantVol(0, self.calendar, self.sigma, self.dayCounter)
        return volatility

    def _set_process(self, ql_stock_price, risk_free_curve, volatility):
        #
        process = ql.BlackScholesProcess(
            ql.QuoteHandle(ql_stock_price),
            ql.YieldTermStructureHandle(risk_free_curve),
            ql.BlackVolTermStructureHandle(volatility))
        return process

    def _set_path_generator(self, steps, process, random_number):
        urng = ql.UniformRandomGenerator(random_number)
        sequenceGenerator = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(steps, urng))
        pathGenerator = ql.GaussianPathGenerator(
            process, self.maturity_time, steps, sequenceGenerator, False)
        return pathGenerator


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

    def _get_schedule(self, start_date, end_date):
        endOfMonth = False
        schedule = ql.Schedule(
            start_date,
            end_date,
            ql.Period(self.trading_time_unit),
            self.calendar,
            ql.ModifiedFollowing,
            ql.ModifiedFollowing,
            ql.DateGeneration.Backward,
            endOfMonth)
        # np.array(self.schedule)
        return schedule

    def _set_evalution_date(self, today: ql.Date):
        if not isinstance(today, ql.Date):
            today = ql.Date(today)
        today = self.calendar.adjust(today)
        ql.Settings.instance().evaluationDate = today
        return today

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


if __name__ == '__main__':
    option_prices = OptionPricer(init_date=ql.Date(18, 11, 2020))
    stock_path_generator = option_prices.create_stock_path_generator()
    stock_price = 100
    new_date = ql.Date(18, 6, 2021)
    # new_date = option_prices.today
    T = option_prices.dayCounter.yearFraction(new_date, option_prices.maturity_date)
    option_prices.calculate_option(stock_price, new_date)
    option_prices.BSM(stock_price, option_prices.strike_price, T, option_prices.risk_free_rate, option_prices.sigma)
    print()
