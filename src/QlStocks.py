import pandas as pd
import numpy as np
import QuantLib as ql
import time
from config.config import RANDOM_SEED
from typing import Union, List

from src.utils import check_list_length
from src.QlCalendar import QlCalendar
from src.QlStock import QlStock

class QlStocks:
    def __init__(
            self,
            ql_calendar: QlCalendar,
            auto_update_with_calendar = True,
    ):
        self.ql_calendar = ql_calendar
        self.update_ql_stock_price = []
        self.ql_calendar.register_callback(self._auto_update_prices)

        self.today_prices = pd.DataFrame(index=pd.Index([], name='codes'))
        self.stock_prices = pd.DataFrame(index=pd.Index([], name='codes'))
        self.auto_update = auto_update_with_calendar

        self.ql_df = pd.DataFrame(
            index=pd.Index([], name='codes'),
            columns=[ 'price_quote', 'dividend_quote', 'volatility', 'process_types', 'processes']
        )
        return

    def stock(self, code) -> QlStock:
        data = self.ql_df.loc[code]
        # price = self.today_prices[code]
        new_stock = QlStock(
            self.ql_calendar,
            code=code,
            price=data['price_quote'],
            dividend=data['dividend_quote'],
            volatility=data['volatility'],
            process_type=data['process_types'],
            process=data['processes'],
        )
        self.update_ql_stock_price.append(new_stock)
        return new_stock

    def add_black_scholes(
            self,
            codes,
            stock_prices: Union[float, ql.SimpleQuote, List[Union[int, float, list, ql.SimpleQuote]]]=100,
            sigma: Union[int, float, List[Union[int, float]]] = 0.2,
            trading_dates=None,
            settlementDays = 0
    ):
        return self._black_scholes_merton(
            codes,
            stock_prices = stock_prices,
            trading_dates = trading_dates,
            sigma = sigma,
            dividend_rates = 0.0,
            settlementDays = settlementDays,
            is_merton = False
        )

    def add_black_scholes_merton(
            self,
            codes=None,
            stock_prices: Union[int, float, ql.SimpleQuote, List[Union[int, float, list, ql.SimpleQuote]]]=100,
            sigma: Union[int, float, List[Union[int, float]]] = 0.2,
            dividend_rates: Union[int, float, List[Union[int, float]]]=0.0,
            trading_dates=None,
            settlementDays=0
    ):
        return self._black_scholes_merton(
            codes,
            stock_prices = stock_prices,
            trading_dates=trading_dates,
            sigma = sigma,
            dividend_rates = dividend_rates,
            settlementDays = settlementDays,
            is_merton = True
        )

    def add_heston(
            self,
            codes=None,
            stock_prices: Union[int, float, ql.SimpleQuote, List[Union[int, float, list, ql.SimpleQuote]]]=100,
            v0: Union[float, List[float]] = 0.005,
            kappa: Union[float, List[float]] = 0.8,
            theta: Union[float, List[float]] = 0.008,
            rho: Union[float, List[float]] = 0.2,
            sigma: Union[float, List[float]] = 0.1,
            dividend_rates: Union[int, float, List[Union[int, float]]] = 0.0,
            trading_dates=None,
            settlementDays=0
    ):
        process_type = 'heston'

        if not (isinstance(codes, list) or isinstance(codes, np.ndarray)):
            codes = [codes]

        codes, stock_prices, v0, kappa, theta, rho, sigma, dividend_rates = check_list_length(
            codes,
            stock_prices,
            v0,
            kappa,
            theta,
            rho,
            sigma,
            dividend_rates
        )

        price_quotes, stock_prices = self._get_stock_price_quotes(stock_prices)

        dividend_rates = [ql.SimpleQuote(float(rate)) for rate in dividend_rates]

        dividend_curve_handle = [
            ql.YieldTermStructureHandle(
                ql.FlatForward(
                    settlementDays,  # settlementDays
                    self.ql_calendar.calendar,
                    ql.QuoteHandle(rate),
                    self.ql_calendar.day_counter
                )
            ) for rate in dividend_rates
        ]

        processes = [
            ql.HestonProcess(
                self.ql_calendar.risk_free_rate_curve_handle,
                dividend,
                ql.QuoteHandle(price),
                v,
                k,
                t,
                s,
                r
            )
            for dividend, price, v, k, t, s, r in
            zip(dividend_curve_handle, price_quotes, v0, kappa, theta, sigma, rho)
        ]

        process_type = [process_type] * len(price_quotes)

        # 添加到DataFrame
        new_df = self._add_to_dataframe(
            codes, price_quotes, dividend_rates, [None] * len(price_quotes), process_type, processes)

        self._concate_stock_prices(stock_prices, codes, trading_dates)

        self._auto_update_prices()

        return new_df

    def get_today_prices(self, codes=None):
        if codes is None:
            #
            # [i.value() for i in self.df['price_quote']]
            return self.today_prices
        else:
            return self.today_prices.loc[codes]

    def set_today_prices(
            self,
            prices:Union[float, np.ndarray, list, pd.DataFrame],
    ):
        if isinstance(prices, list) or isinstance(prices, np.ndarray):
            if len(prices) != len(self.stock_prices):
                raise 'wrong prices length'
            else:
                self.stock_prices[self.ql_calendar.today()] = prices
                self.today_prices = self.stock_prices[self.ql_calendar.today()]
        elif isinstance(prices, pd.DataFrame):
            self.stock_prices[self.ql_calendar.today()] = prices[self.ql_calendar.today()]
            self.today_prices = prices[self.ql_calendar.today()]
        elif isinstance(prices, int):
            prices = float(prices)
            self.stock_prices[self.ql_calendar.today()] = prices
            self.today_prices = self.stock_prices[self.ql_calendar.today()]
        elif isinstance(prices, float):
            self.stock_prices[self.ql_calendar.today()] = prices
            self.today_prices = self.stock_prices[self.ql_calendar.today()]

        [p.setValue(new_p) for p, new_p in zip(self.ql_df['price_quote'].values, self.today_prices.values)]
        return

    def set_all_prices(
            self,
            stock_prices: Union[np.ndarray, list, pd.DataFrame],
            dates=None,
    ):
        if len(self.stock_prices) != len(stock_prices):
            raise 'wrong prices length'

        if isinstance(stock_prices, pd.DataFrame):
            self.stock_prices = stock_prices
        else:
            if dates is None:
                dates = self._get_trading_dates(len(stock_prices[0]))
            self.stock_prices = pd.DataFrame(stock_prices, index=self.stock_prices.index, columns=dates)

        self._auto_update_prices()

        return self.stock_prices

    # def set_one_day_prices(self, prices, df=None):
    #
    #     if df is None:
    #         df = self.df
    #     if isinstance(prices, float) or isinstance(prices, int):
    #         prices = [prices] * len(df)
    #     [p.setValue(new_p) for p, new_p in zip(df['price_quote'].values, prices)]
    #     # self.df['price_quote'].values[0].value()
    #     # [i.value() for i in self.df['price_quote'].values]
    #     return

    def _auto_update_prices(self):
        if self.auto_update:
            try:
                self.today_prices = self.stock_prices[self.ql_calendar.today()]
            except:
                # print(f'At {self.ql_calendar.today()}, '
                #       f'stocks do not have prices data, '
                #       f'please use function - < set_today_prices > to set today prices')
                # self.stock_prices[self.ql_calendar.today()] = np.nan
                # self.today_prices = self.stock_prices[self.ql_calendar.today()]
                return
            [p.setValue(new_p) for p, new_p in zip(self.ql_df['price_quote'].values, self.today_prices.values)]
            return

    def _add_to_dataframe(self, code, stock_price, dividend_rate, volatility, process_type, process):
        """添加数据到DataFrame"""
        new_df = pd.DataFrame(
            {
                'price_quote': stock_price,
                'dividend_quote': dividend_rate,
                'volatility': volatility,
                'process_types': process_type,
                'processes': process,
            },
            index = pd.Index(code, name='codes')
        )
        self.ql_df = pd.concat([self.ql_df, new_df], verify_integrity=True)
        return new_df

    def _create_dividend_curve(
            self,
            dividend_rates: List[ql.SimpleQuote],
            settlementDays = 0
    ):
        """创建股息曲线处理"""

        handles = [
            ql.YieldTermStructureHandle(
                ql.FlatForward(
                    settlementDays,  # settlementDays
                    self.ql_calendar.calendar,
                    ql.QuoteHandle(rate),
                    self.ql_calendar.day_counter
                )
            ) for rate in dividend_rates
        ]
        return handles

    def _black_scholes_merton(
            self,
            codes,
            stock_prices: Union[int, float, ql.SimpleQuote, List[Union[int, float, list, ql.SimpleQuote]]] = 100,
            trading_dates=None,
            sigma: Union[int, float, List[Union[int, float]]] = 0.2,
            dividend_rates: Union[int, float, List[Union[int, float]]] = 0.0,
            settlementDays=0,
            is_merton=False,
    ):
        if not (isinstance(codes, list) or isinstance(codes, np.ndarray)):
            codes = [codes]

        codes, stock_prices, sigma, dividend_rates = check_list_length(codes, stock_prices, sigma, dividend_rates)

        price_quotes, stock_prices = self._get_stock_price_quotes(stock_prices)

        sigma = [ql.SimpleQuote(float(s)) for s in sigma]

        volatility_handle = [
            ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(
                    settlementDays,
                    self.ql_calendar.calendar,
                    ql.QuoteHandle(s),
                    self.ql_calendar.day_counter)
            ) for s in sigma
        ]

        if is_merton:
            process_type = 'black_schole_merton'

            dividend_rates = [ql.SimpleQuote(float(rate)) for rate in dividend_rates]

            dividend_curve_handle = [
                ql.YieldTermStructureHandle(
                    ql.FlatForward(
                        settlementDays,  # settlementDays
                        self.ql_calendar.calendar,
                        ql.QuoteHandle(rate),
                        self.ql_calendar.day_counter
                    )
                ) for rate in dividend_rates
            ]

            process = [
                ql.BlackScholesMertonProcess(
                    ql.QuoteHandle(price),
                    dividend,
                    self.ql_calendar.risk_free_rate_curve_handle,
                    vol
                ) for price, dividend, vol in zip(price_quotes, dividend_curve_handle, volatility_handle)
            ]

        else:
            process_type = 'black_schole'
            dividend_rates = [None] * len(price_quotes)

            process = [
                ql.BlackScholesProcess(
                    ql.QuoteHandle(price),
                    self.ql_calendar.risk_free_rate_curve_handle,
                    vol
                ) for price, vol in zip(price_quotes, volatility_handle)
            ]

        process_type = [process_type] * len(price_quotes)

        # 添加到DataFrame
        new_df = self._add_to_dataframe(codes, price_quotes, dividend_rates, sigma, process_type, process)

        # prices_df = pd.DataFrame(stock_prices, index=new_df['codes'].values)

        self._concate_stock_prices(stock_prices, new_df.index.values, trading_dates)
        # self.stock_prices = pd.concat([self.stock_prices, prices_df])
        self._auto_update_prices()
        return new_df

    def _concate_stock_prices(self, prices, codes, dates=None):
        if dates is None:
            dates = self._get_trading_dates(len(prices[0]))

        prices_df = pd.DataFrame(prices, index=codes, columns=dates)

        self.stock_prices = pd.concat([self.stock_prices, prices_df])

        return prices_df

    def _get_trading_dates(self, len_days):

        today = self.ql_calendar.today()
        all_dates = self.ql_calendar.all_trading_dates
        all_dates = all_dates[all_dates >= today]

        if len(all_dates) < len_days:

            self.ql_calendar.all_trading_dates = np.concatenate((
                all_dates,

                self.ql_calendar.get_trading_dates(
                    self.ql_calendar.all_trading_dates[-1],
                    end=len_days
                )))
        dates = self.ql_calendar.all_trading_dates[self.ql_calendar.all_trading_dates >= today][:len_days]

        return dates

    def _get_stock_price_quotes(self, stock_prices):
        stock_prices = np.array(stock_prices)
        if stock_prices.ndim == 1:
            stock_prices = stock_prices[:, np.newaxis]

        if not isinstance(stock_prices[0][0], ql.SimpleQuote):
            stock_prices = np.array(stock_prices, dtype=np.float64)
            price_quote = [
                ql.SimpleQuote(float(price[0])) for price in stock_prices
            ]

        else:
            price_quote = [i[0] for i in stock_prices]
            stock_prices = np.array([i.value() for i in price_quote])[:, np.newaxis]
        return price_quote, stock_prices


if __name__ == '__main__':
    start_date = ql.Date(1,1,2023)

    ql_calendar = QlCalendar(init_date=start_date)

    s = QlStocks(ql_calendar)

    stocks = [f'bsm_{i}' for i in range(2)]
    # stocks= 'aaa'

    prices0 = np.random.normal(loc=100, scale=10.0, size=[len(stocks), 5])
    # prices = 50

    s.add_black_scholes_merton(stocks, prices0)

    prices0 = np.random.normal(loc=100, scale=10.0, size=[len(stocks), 100])
    # prices = 50
    stocks = [f'bsm_{i + 2}' for i in range(2)]
    dates = ql_calendar.get_trading_dates(start_date + 8, 100)
    s.add_black_scholes_merton(stocks, prices0, trading_dates=dates)
    #
    # stocks = [f'bs_{i}' for i in range(2)]
    # #
    # s.black_scholes(stocks, [[100.0, 101.1], [50.0, 51.1]])

    ql_calendar.to_next_trading_date()

    new_df = s.add_heston(
        codes = ['h_0', 'h_1'],
        stock_prices = [100.0] * 2,
        v0=0.04, kappa=1.0, theta=0.06, rho=-0.3, sigma=0.4, dividend_rates=0.01)
    # s.path_generators(date_param=50)

    # stock_path_generator = s.stock_path_generator(
    #     date_param=252,
    #     process=s.ql_df.iloc[0]['processes'],
    # )
    # t0 = time.time()
    # [np.array(stock_path_generator.next().value()) for i in range(1000)]
    # print(time.time() - t0)
    #
    #
    # t0 = time.time()
    # prices0 = s.stock_paths(100, 252, process=s.ql_df.iloc[0]['processes'])
    # print(time.time() - t0)



    print()
