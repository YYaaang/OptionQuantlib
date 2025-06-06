import QuantLib as ql
import numpy as np
import pandas as pd
from QuantLib import PlainVanillaPayoff
from typing import Optional, Union, List
import time

from tables import all_types

from src.QlCalendar import QlCalendar
from src.QlStocks import QlStocks



class QlVanillaOptions():
    """ """
    def __init__(
        self,
        stock_df: Union[pd.Series, pd.DataFrame],
    ):
        if isinstance(stock_df, pd.Series):
            df = stock_df[['codes', 'price_quote', 'processes']].to_frame()
        elif isinstance(stock_df, pd.DataFrame):
            df = stock_df[['codes', 'price_quote', 'processes']]
        else:
            raise ValueError('wrong stock_df type')
        self.stocks_df = df.copy()
        columns = ['codes', 'price_quote', 'processes', 'strike_prices', 'maturity_dates',
                   'types', 'payoff', 'maturity', 'engines']
        self.options_df = pd.DataFrame(columns=columns)
        return

    def european_option(
            self,
            option_types: Union[str, List[str], np.ndarray],
            strike_prices: Union[float, List[float], np.ndarray],
            maturity_dates: Union[ql.Date, List[ql.Date], np.ndarray],
            qlExercise:ql.Exercise = ql.EuropeanExercise,
            qlEngine:ql.PricingEngine = ql.AnalyticEuropeanEngine,
            *args,
            **kwargs
    ):
        if isinstance(strike_prices, float):
            strike_prices = [strike_prices]

        params = {'option_types': option_types, 'strike_prices': strike_prices, 'maturity_dates': maturity_dates}
        param_lists = {k: v for k, v in params.items() if isinstance(v, list) or isinstance(v, np.ndarray)}

        # 确保所有列表参数长度一致
        lengths = [len(v) for v in param_lists.values()]
        if len(lengths) == 0:
            size = 1
        else:
            size = lengths[0]
        if len(set(lengths)) > 1:
            raise ValueError("所有列表参数(option_types, strike_prices, maturity_dates)长度必须一致")

        else:
            n = size
            option_types = option_types if (isinstance(option_types, list) or isinstance(option_types, np.ndarray)) else [option_types] * n
            strike_prices = strike_prices if (isinstance(strike_prices, list) or isinstance(strike_prices, np.ndarray)) else [strike_prices] * n

            if (isinstance(maturity_dates, list) or isinstance(maturity_dates, np.ndarray)):
                maturity = [qlExercise(i) for i in maturity_dates]
            else:
                maturity = [qlExercise(maturity_dates)] * n
                maturity_dates = [maturity_dates] * n

        all_types = [ql.Option.Call if i =='call' else ql.Option.Put for i in option_types]

        payoff: list[PlainVanillaPayoff] = [ql.PlainVanillaPayoff(o, s) for o, s in zip(all_types, strike_prices)]

        new_rows = [
            {'strike_prices': s, 'maturity_dates': mdt, 'types': t, 'payoff': p, 'maturity': mtr}
            for s, mdt, t, p, mtr in zip(strike_prices, maturity_dates, option_types, payoff, maturity)
        ]
        new_df = pd.DataFrame(new_rows)

        # 为df_10和df_3添加临时键列
        self.stocks_df.loc[:, 'key'] = 1
        new_df.loc[:, 'key'] = 1
        new_df = pd.merge(self.stocks_df, new_df, on='key').drop('key', axis=1)

        options = [ql.EuropeanOption(p, e) for p, e in zip(new_df['payoff'].values, new_df['maturity'].values)]

        engines = [qlEngine(i, *args, **kwargs) for i in new_df['processes'].values]
        [option.setPricingEngine(engine) for option, engine in zip(options, engines)]

        new_df['engines'] = engines
        new_df['options'] = options
        self.options_df = pd.concat([self.options_df, new_df], ignore_index=True)

        return new_df

    def NPV(self, df=None):
        if df is None:
            df = self.options_df.copy()
        if isinstance(df, pd.Series):
            df['NPV'] = df['options'].NPV()
        else:
            df['NPV'] = [i.NPV() for i in df['options']]
            # df.iloc[0]['options']
        return df

    def delta(self, df=None):
        if df is None:
            df = self.options_df.copy()
        df['delta'] = [i.delta() for i in df['options']]
        return df

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

if __name__ == '__main__':
    start_date = ql.Date(1,1,2023)

    ql_calendar = QlCalendar(init_date=start_date)

    s = QlStocks(ql_calendar)

    s.black_scholes([100.0, 50.0])

    options = QlVanillaOptions(s.df)
    #
    prices = 100
    options.european_option(
        'put',
        prices,
        ql.Date(3,2,2023)
    )

    prices = np.random.normal(loc=100, scale=10, size=5000)
    # prices = 100
    options.european_option(
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