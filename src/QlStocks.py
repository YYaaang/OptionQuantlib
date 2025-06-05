import pandas as pd
import numpy as np
import QuantLib as ql
import time
from config.config import RANDOM_SEED
from src.QlCalendar import QlCalendar
from typing import Union, List


class QlStocks:
    def __init__(
            self,
            ql_calendar: QlCalendar,
            prices: Union[list, np.array] = [],
            codes: Union[list, np.array] = None,
    ):
        self.ql_calendar = ql_calendar
        self.today = self.ql_calendar.today
        # columns = ['codes', 'price_quote', 'processes']
        if codes is None:
            codes = [f'tmp_{i}' for i in np.arange(len(prices))]
        df = pd.DataFrame(codes, columns=['codes'])

        df['price_quote'] = [ql.SimpleQuote(i) for i in prices]
        df['dividend_quote'] = None
        df['process_types'] = None
        df['processes'] = None
        self.df = df
        return

    def black_scholes(
            self,
            stock_price: Union[int, float, ql.SimpleQuote, List[Union[int, float, ql.SimpleQuote]]],
            sigma: Union[int, float, List[Union[int, float]]] = 0.2,
            code = None):

        process_type = 'black_schole'
        stock_price = self._check(stock_price)

        # 验证输入
        self._validate_inputs(stock_price, sigma, code)

        # 创建波动率处理
        volatility_handle = self._create_volatility_handle(sigma)
        if isinstance(stock_price, list) and not isinstance(volatility_handle, list):
            volatility_handle = [volatility_handle] * len(stock_price)

        # 创建process
        if isinstance(stock_price, list):
            process = [
                ql.BlackScholesProcess(
                    ql.QuoteHandle(price),
                    self.ql_calendar.risk_free_rate_curve_handle,
                    vol
                ) for price, vol in zip(stock_price, volatility_handle)
            ]
            dividend_rate = [None] * len(stock_price)

        else:
            process = ql.BlackScholesProcess(
                ql.QuoteHandle(stock_price),
                self.ql_calendar.risk_free_rate_curve_handle,
                volatility_handle
            )
            dividend_rate = None
        # 添加到DataFrame
        new_df = self._add_to_dataframe(code, stock_price, dividend_rate, process_type, process)
        return new_df

    def black_scholes_merton(
            self,
            stock_price: Union[int, float, ql.SimpleQuote, List[Union[int, float, ql.SimpleQuote]]],
            sigma: Union[int, float, List[Union[int, float]]] = 0.2,
            dividend_rate: Union[int, float, List[Union[int, float]]]=None,
            code=None):

        process_type = 'black_scholes_merton'
        stock_price = self._check(stock_price)

        # 验证输入
        self._validate_inputs(stock_price, sigma, code)

        # 创建波动率处理
        volatility_handle = self._create_volatility_handle(sigma)
        if isinstance(stock_price, list) and not isinstance(volatility_handle, list):
            volatility_handle = [volatility_handle] * len(stock_price)

        # 创建股息曲线处理
        dividend_rate, dividend_curve_handle = self._create_dividend_curve(dividend_rate)
        if isinstance(stock_price, list) and not isinstance(dividend_curve_handle, list):
            dividend_curve_handle = [dividend_curve_handle] * len(stock_price)

        # 创建process
        if isinstance(stock_price, list):
            process = [
                ql.BlackScholesMertonProcess(
                    ql.QuoteHandle(price),
                    dividend,
                    self.ql_calendar.risk_free_rate_curve_handle,
                    vol
                ) for price, dividend, vol in zip(stock_price, dividend_curve_handle, volatility_handle)
            ]
        else:
            process = ql.BlackScholesMertonProcess(
                ql.QuoteHandle(stock_price),
                dividend_curve_handle,
                self.ql_calendar.risk_free_rate_curve_handle,
                volatility_handle
            )

        # 添加到DataFrame
        new_df = self._add_to_dataframe(code, stock_price, dividend_rate, process_type, process)
        return new_df

    def heston(
            self,
            stock_price: Union[int, float, ql.SimpleQuote, List[Union[int, float, ql.SimpleQuote]]],
            v0: Union[float, List[float]] = 0.005,
            kappa: Union[float, List[float]] = 0.8,
            theta: Union[float, List[float]] = 0.008,
            rho: Union[float, List[float]] = 0.2,
            sigma: Union[float, List[float]] = 0.1,
            dividend_rate: Union[int, float, List[Union[int, float]]] = 0.0,
            code=None):

        process_type = 'heston'
        stock_price = self._check(stock_price)

        # 验证输入
        self._validate_inputs(stock_price, code=code)

        # 检查参数长度一致性
        params = {'v0': v0, 'kappa': kappa, 'theta': theta, 'rho': rho, 'sigma': sigma, 'dividend_rate': dividend_rate}
        param_lists = {k: v for k, v in params.items() if isinstance(v, list)}

        if param_lists:
            # 确保所有列表参数长度一致
            lengths = [len(v) for v in param_lists.values()]
            if len(set(lengths)) > 1:
                raise ValueError("所有列表参数(v0, kappa, theta, rho, sigma)长度必须一致")

            # 确保与stock_price长度一致(如果是list)
            if isinstance(stock_price, list) and lengths[0] != len(stock_price):
                raise ValueError("当stock_price为list时，参数列表长度必须与stock_price一致")

        # 创建股息曲线处理
        dividend_rate, dividend_curve_handle = self._create_dividend_curve(dividend_rate)

        if isinstance(stock_price, list):
            v0 = v0 if isinstance(v0, list) else [v0] * len(stock_price)
            kappa = kappa if isinstance(kappa, list) else [kappa] * len(stock_price)
            theta = theta if isinstance(theta, list) else [theta] * len(stock_price)
            rho = rho if isinstance(rho, list) else [rho] * len(stock_price)
            sigma = sigma if isinstance(sigma, list) else [sigma] * len(stock_price)
            dividend_rate = dividend_rate if isinstance(dividend_rate, list) else [dividend_rate] * len(stock_price)
            dividend_curve_handle = dividend_curve_handle if isinstance(dividend_curve_handle, list) \
                else [dividend_curve_handle] * len(stock_price)

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
                zip(dividend_curve_handle, stock_price, v0, kappa, theta, sigma, rho)
            ]
            new_df = self._add_to_dataframe(code, stock_price, dividend_rate, process_type, processes)

        else:
            # 处理单个创建
            process = ql.HestonProcess(
                self.ql_calendar.risk_free_rate_curve_handle,
                dividend_curve_handle,
                ql.QuoteHandle(stock_price),
                v0,
                kappa,
                theta,
                sigma,
                rho
            )

            # 添加到DataFrame
            new_df = self._add_to_dataframe(code, stock_price, dividend_rate, process_type, process)
        return new_df

    def stock_paths(
            self,
            paths,
            date_param: [int, ql.Date],  # timesteps, end_date
            process,
            random_seed=RANDOM_SEED
    ):
        stock_path_generator = self.stock_path_generator(
            date_param = date_param,
            process = process,
            random_seed = random_seed
        )
        n = process.factors()
        if n == 1:
            stock_paths = np.array([np.array(stock_path_generator.next().value()) for _ in np.arange(paths)])
        else:
            stock_paths = np.array([np.array(stock_path_generator.next().value()[0]) for _ in np.arange(paths)])
        return stock_paths

    def stock_path_generator(
            self,
            date_param:[int, ql.Date], #  timesteps, end_date
            process,
            random_seed=RANDOM_SEED
    ):
        timesteps, total_time = self._set_date_parm(date_param)

        # 创建路径生成器
        stock_path_generator = self._set_path_generator(total_time, timesteps, process, random_seed=random_seed)
        # 生成1条路径
        # stock_path = stock_path_generator.next().value()
        return stock_path_generator


    # def path_generators(
    #         self,
    #         df=None,
    #         date_param: [int, ql.Date] = 1,  # timesteps, end_date
    #         random_seed = RANDOM_SEED
    # ):
    #     if df is None:
    #         df = self.df.copy()
    #     processes = df['processes'].values
    #
    #     timesteps, total_time = self._set_date_parm(date_param)
    #
    #     path_generator_lists = [
    #         self._set_path_generator(total_time, timesteps, process, random_seed=i) for i, process in enumerate(processes)
    #     ]
    #
    #     paths = [generator.next().value() for generator in path_generator_lists]
    #
    #     paths_array = np.array(paths)
    #     return paths_array
    #
    #     path_generator_lists = [
    #         self._set_path_generator(total_time, timesteps, process, random_seed=random_seed) for process in processes
    #     ]
    #     df['path_generators'] = path_generator_lists
    #
    #     # new_prices_array = [np.array(generator.next().value()) for generator in path_generator_lists]
    #     # new_prices_array = np.array([i if i.ndim==1 else i[0] for i in new_prices_array])
    #     # # path_generator_lists[2].next().value()
    #     # new_prices_array[:, 1:]
    #
    #     return df

    def set_prices(self, prices, df=None):
        if df is None:
            df = self.df
        if isinstance(prices, float) or isinstance(prices, int):
            prices = [prices] * len(df)
        [p.setValue(new_p) for p, new_p in zip(df['price_quote'].values, prices)]
        # self.df['price_quote'].values[0].value()
        # [i.value() for i in self.df['price_quote'].values]
        return

    def _set_path_generator(self, length, timesteps, process, random_seed=RANDOM_SEED):
        #
        pathGenerator = None
        if type(process) in [ql.BlackScholesMertonProcess, ql.BlackScholesProcess]:
            urng = ql.UniformRandomGenerator(random_seed)
            sequenceGenerator = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(timesteps, urng))
            pathGenerator = ql.GaussianPathGenerator(
                process, length, timesteps, sequenceGenerator, False)
        elif type(process) is ql.HestonProcess:
            dimension = process.factors()
            rng = ql.UniformLowDiscrepancySequenceGenerator(dimension * timesteps, random_seed)
            sequenceGenerator = ql.GaussianLowDiscrepancySequenceGenerator(rng)
            time_grid = ql.TimeGrid(length, timesteps)
            pathGenerator = ql.GaussianSobolMultiPathGenerator(
                process, time_grid, sequenceGenerator, False)

        # np.array(pathGenerator.next().value())
        return pathGenerator

    def _set_date_parm(self, date_param):

        if isinstance(date_param, int):
            timesteps = date_param
            end_date = self.ql_calendar.cal_date_advance(init_date=self.today, times=timesteps, time_unit='days')
            print(f"使用步数: {timesteps} 步")
        elif isinstance(date_param, ql.Date):
            end_date = self.ql_calendar._set_evalution_date(date_param)
            timesteps = self.ql_calendar.day_counter.dayCount(self.today, end_date)
            print(f"使用结束日期: {end_date}")
        else:
            raise TypeError("参数必须是 float（total_length）或 ql.Date（end_date）")

        total_time = self.ql_calendar.day_counter.yearFraction(self.today, end_date)

        print(f'timesteps: {timesteps}, Time length(per year): {total_time}'
              f'start_date: {self.today} end_date: {end_date}')

        return timesteps, total_time


    def _check(self, stock_price):
        if isinstance(stock_price, int):
            stock_price = ql.SimpleQuote(float(stock_price))
        elif isinstance(stock_price, float):
            stock_price = ql.SimpleQuote(stock_price)
        elif isinstance(stock_price, list):
            stock_price = [self._check(price) for price in stock_price]
        return stock_price

    def _add_to_dataframe(self, code, stock_price, dividend_rate, process_type, process):
        """添加数据到DataFrame"""
        if isinstance(stock_price, list):
            code = code or [f'tmp_{len(self.df) + 1 + i}' for i in range(len(stock_price))]
            new_rows = [
                {'codes': c, 'price_quote': s, 'dividend_quote': d, 'process_types': process_type, 'processes': pr}
                for c, s, d, pr in zip(code, stock_price, dividend_rate, process)
            ]
            new_df = pd.DataFrame(new_rows)
            self.df = pd.concat([self.df, new_df], ignore_index=True)
        else:
            code = code or f'tmp_{len(self.df) + 1}'
            self.df.loc[len(self.df)] = [code, stock_price, dividend_rate, process_type, process]
            new_df = pd.DataFrame([[code, stock_price, dividend_rate, process_type, process]],
                                  columns=self.df.columns)
        return new_df

    def _validate_inputs(self, stock_price, sigma=None, code=None):
        """验证输入参数类型和长度是否匹配"""
        if (not isinstance(stock_price, list)) and isinstance(sigma, list):
            raise ValueError('单个stock_price时，sigma不能是list')

        if isinstance(stock_price, list) and isinstance(sigma, list):
            if len(stock_price) != len(sigma):
                raise ValueError('当stock_price和sigma都为list时，长度必须相同')
            if code is not None and len(stock_price) != len(code):
                raise ValueError('stock_price和code必须长度相同')

    def _create_volatility_handle(self, sigma: Union[float, List[float], None]):
        """创建波动率处理"""
        settlementDays = 0

        if isinstance(sigma, list):
            return [
                ql.BlackVolTermStructureHandle(
                    ql.BlackConstantVol(settlementDays, self.ql_calendar.calendar, s, self.ql_calendar.day_counter)
                ) for s in sigma
            ]
        else:
            volatility = ql.BlackConstantVol(
                settlementDays, self.ql_calendar.calendar, sigma, self.ql_calendar.day_counter)
            return ql.BlackVolTermStructureHandle(volatility)

    def _create_dividend_curve(
            self,
            dividend_rate: Union[float, List[float]]):
        """创建股息曲线处理"""
        settlementDays = 0

        dividend_rate = self._check(dividend_rate)
        if isinstance(dividend_rate, list):
            return dividend_rate, [
                ql.YieldTermStructureHandle(
                    ql.FlatForward(
                        settlementDays,  # settlementDays
                        self.ql_calendar.calendar,
                        ql.QuoteHandle(rate),
                        self.ql_calendar.day_counter
                    )
                ) for rate in dividend_rate
            ]
        else:
            return dividend_rate, ql.YieldTermStructureHandle(
                ql.FlatForward(
                    settlementDays,  # settlementDays
                    self.ql_calendar.calendar,
                    ql.QuoteHandle(dividend_rate),
                    self.ql_calendar.day_counter
                )
            )


if __name__ == '__main__':
    start_date = ql.Date(1,1,2023)

    ql_calendar = QlCalendar(init_date=start_date)

    s = QlStocks(ql_calendar)

    # s.black_scholes(100.0)
    # s.black_scholes([100.0] * 1000)

    # s.path_generators(date_param=50)

    #
    # s.black_scholes([100.0, 50.0])
    new_df = s.heston([100.0] * 2, v0=0.04, kappa=1.0, theta=0.06, rho=-0.3, sigma=0.4, dividend_rate=0.01)
    # s.path_generators(date_param=50)

    s.stock_paths(200, 200, process=s.df.loc[0, 'processes'])



    print()
