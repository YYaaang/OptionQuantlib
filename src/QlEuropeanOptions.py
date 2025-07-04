import QuantLib as ql
import numpy as np
import pandas as pd
from QuantLib import PlainVanillaPayoff
from typing import Optional, Union, List
import time

from src.utils import check_list_length, set_analytic_engine, set_fd_engine, set_option_name
from src.QlCalendar import QlCalendar
from src.QlStock import QlStock
from src.QlEuropeanOption import QlEuropeanOption

class QlEuropeanOptions:
    """ """
    def __init__(
            self,
            ql_stock: QlStock,
    ):
        self.ql_calendar = ql_stock.ql_calendar
        self.stock = ql_stock

        self.columns = ['payoff', 'exercise', 'options']

        self.options_df_index = ['codes', 'types', 'strike', 'maturity']

        self.options_df = pd.DataFrame(
            index=pd.MultiIndex.from_arrays(
                [[]] * len(self.options_df_index),
                names = self.options_df_index
            ),
            columns=self.columns
        )

        return

    def all_options(self):
        return self.options_df

    def option(self, code) -> QlEuropeanOption:

        data = self.options_df.loc[code]
        data.reset_index(inplace=True)
        data = data.iloc[0]
        #
        ql_option:QlEuropeanOption = QlEuropeanOption(
            code = code,
            option_type = data['types'],
            strike = data['strike'],
            maturity = data['maturity'],
            option = data['options'],
            stock_price_quote = self.stock.price_quote,
            process_type = self.stock.process_type,
            process = self.stock.process,
        )
        ql_option.stock_vol_quote = self.stock.volatility
        #
        return ql_option

    def add_options(
            self,
            option_types: Union[str, List[str], np.ndarray] = None,
            strike_prices: Union[float, List[float], np.ndarray] = None,
            maturity_dates: Union[ql.Date, List[ql.Date], np.ndarray] = None,
            # engine_types: Union[str, List[str], np.ndarray] = None, # 'Analytic', 'Fd', 'MC'
            codes=None,
            qlExercise: ql.Exercise = ql.EuropeanExercise,
    ):
        strike_prices, maturity_dates, option_types, codes = check_list_length(
            strike_prices, maturity_dates, option_types, codes)

        new_df = self._payoff_and_exercise(codes, strike_prices, option_types, maturity_dates, qlExercise)

        options = [ql.EuropeanOption(p, e) for p, e in zip(new_df['payoff'].values, new_df['exercise'].values)]

        new_df['options'] = options
        # new_df['engines'] = None

        new_df.set_index(self.options_df_index, inplace=True)
        self.options_df = pd.concat([self.options_df, new_df])
        #
        return new_df['options']

    def analytic_engines(
            self,
            codes: Union[list, np.ndarray, pd.DataFrame] = None,
    ):
        options = self._codes_get_options(codes)
        #
        engine = set_analytic_engine(self.stock.process_type, self.stock.process)

        [option.setPricingEngine(engine) for option in options]
        # self.options_df.loc[index, 'engines'] = engine
        #
        return


    def fd_engines(
            self,
            codes: Union[str, list, np.ndarray, pd.DataFrame] = None,
            tGrid = 100,
            xGrid = 100,
            vGrid = 50

    ):
        options = self._codes_get_options(codes)
        #
        engine = set_fd_engine(self.stock.process_type, self.stock.process, tGrid, xGrid, vGrid)
        [option.setPricingEngine(engine) for option in options]

        # self.options_df.loc[index, 'engines'] = engine
        #
        return


    # def european_option(
    #         self,
    #         option_types: Union[str, List[str], np.ndarray],
    #         strike_prices: Union[float, List[float], np.ndarray],
    #         maturity_dates: Union[ql.Date, List[ql.Date], np.ndarray],
    #         qlExercise:ql.Exercise = ql.EuropeanExercise,
    #         qlEngine:ql.PricingEngine = ql.AnalyticEuropeanEngine,
    #         *args,
    #         **kwargs
    # ):
    #     """
    #     需改进
    #     :param option_types:
    #     :param strike_prices:
    #     :param maturity_dates:
    #     :param qlExercise:
    #     :param qlEngine:
    #     :param args:
    #     :param kwargs:
    #     :return:
    #     """
    #     new_df = self._check_param_length(strike_prices, option_types, maturity_dates, qlExercise)
    #
    #     # 为df_10和df_3添加临时键列
    #     stock_df = self.ql_stocks.ql_df.copy()
    #     stock_df.loc[:, 'key'] = 1
    #     new_df.loc[:, 'key'] = 1
    #     new_df = pd.merge(stock_df, new_df, on='key').drop('key', axis=1)
    #
    #     # calculate options
    #     options = [ql.EuropeanOption(p, e) for p, e in zip(new_df['payoff'].values, new_df['exercise'].values)]
    #
    #     engines = [qlEngine(i, *args, **kwargs) for i in new_df['processes'].values]
    #     [option.setPricingEngine(engine) for option, engine in zip(options, engines)]
    #
    #     new_df['engines'] = engines
    #     new_df['options'] = options
    #     self.options_df = pd.concat([self.options_df, new_df], ignore_index=True)
    #
    #     return new_df

    def impliedVolatility(self, NPVs):
        return [self._impliedVolatility(option, npv) for option, npv in zip(self.options_df['options'], NPVs)]

    def NPV(self, options: [pd.Series, pd.DataFrame] = None):

        if options is None:
            options = self.options_df['options'].values
        elif isinstance(options, pd.Series):
            options = options.values
        elif isinstance(options, pd.DataFrame):
            options = options['options'].values

        return np.array([i.NPV() for i in options])

    def delta(
            self,
            options: pd.Series = None,
    ):
        if options is None:
            options = self.options_df['options'].values
        elif isinstance(options, pd.Series):
            options = options.values
        elif isinstance(options, pd.DataFrame):
            options = options['options'].values

        try:
            delta = np.array([i.delta() for i in options])
        except:
            raise 'current option type do not have Quantlib default delta, please use numerical_differentiation=True to calculate'
        return delta


    def delta_numerical(
            self,
            options: pd.Series = None,
            h=0.01
    ):
        if options is None:
            options = self.options_df['options'].values
        elif isinstance(options, pd.Series):
            options = options.values
        elif isinstance(options, pd.DataFrame):
            options = options['options'].values

        price_quotes = self.stock.price_quote
        price_value = self.stock.price_quote.value()

        price_quotes.setValue(price_value + h)
        npv_up = np.array([i.NPV() for i in options])

        price_quotes.setValue(price_value - h)
        npv_down = np.array([i.NPV() for i in options])

        delta = (npv_up - npv_down) / (2 * h)

        price_quotes.setValue(price_value)

        return delta



    def gamma(self, df=None):
        if df is None:
            df = self.options_df.copy()
        df['gamma'] = [i.gamma() for i in df['options']]
        return df

    def vega(self, df=None):
        if df is None:
            df = self.options_df.copy()
        df['vega'] = [i.vega() for i in df['options']]
        return df

    def _payoff_and_exercise(self, codes, strike_prices, option_types, maturity_dates, qlExercise):

        all_types = [ql.Option.Call if i =='call' else ql.Option.Put for i in option_types]
        all_codes = [code
                 if code is not None else
                 set_option_name(self.stock.code, date, strike, option_type)
                 for code, date, strike, option_type in zip(codes, maturity_dates, strike_prices, all_types)]


        payoff: list[PlainVanillaPayoff] = [ql.PlainVanillaPayoff(o, s) for o, s in zip(all_types, strike_prices)]
        exercise = [qlExercise(i) for i in maturity_dates]

        new_df = pd.DataFrame(
            {'codes': all_codes,
            'strike': strike_prices,
             'maturity': maturity_dates,
             'types': option_types,
             'payoff': payoff,
             'exercise': exercise}
        )
        return new_df

    def _codes_get_options(self, codes:Union[list, np.ndarray, pd.DataFrame]=None):
        if codes is None:
            return self.options_df['options'].values
        elif isinstance(codes, (list, np.ndarray)):
            return self.options_df.loc[codes, 'options'].values
        elif isinstance(codes, pd.DataFrame):
            return codes['options'].values

    def _impliedVolatility(self, option, npv):
        try:
            return option.impliedVolatility(npv, self.stock.process)
        except:
            # print('fail calculate impliedVolatility')
            return np.nan

if __name__ == '__main__':
    from src.QlStocks import QlStocks
    start_date = ql.Date(1,1,2023)

    ql_calendar = QlCalendar(init_date=start_date)

    s = QlStocks(ql_calendar)

    s.add_black_scholes(codes='s_1', stock_prices = [[100.0, 110.0, 111.0, 101.0, 99]])
    s_1 = s.stock('s_1')

    options = QlEuropeanOptions(s_1)

    #
    prices = 100
    options.add_options(
        'put',
        prices,
        ql.Date(3,2,2023),
    )
    options.analytic_engines()
    put = options.option('s_1230203P00100000')
    put.analytic_engine()

    prices = np.random.normal(loc=100, scale=10, size=5000)
    # prices = 100
    options.add_options(
        'call',
        prices,
        ql.Date(3,2,2023)
    )
    df = options.options_df
    options.NPV(df)
    options.delta(df)
    options.gamma(df)
    options.vega(df)
    print()