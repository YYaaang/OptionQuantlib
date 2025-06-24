import QuantLib as ql
import numpy as np
from src.utils import set_analytic_engine, set_fd_engine

class QlEuropeanOption:
    def __init__(
            self,
            code,
            option_type,
            strike: float,
            maturity: ql.Date,
            option: ql.EuropeanOption,
            stock_price_quote: ql.SimpleQuote,
            process_type,
            process,
    ):
        """
        初始化欧式期权对象
        :param payoff: 期权收益类型（Call/Put）
        :param exercise: 行权日期
        """
        self.code = code
        self.option_type = option_type
        self.strike = strike
        self.maturity_date = maturity
        self.option = option

        self.stock_price_quote = stock_price_quote
        self.process_type = process_type
        self.process = process

        self.stock_vol_quote = None
        return

    def analytic_engine(self):
        engine = set_analytic_engine(self.process_type, self.process)
        self.option.setPricingEngine(engine)
        return

    def fd_engine(
            self,
            tGrid=100,
            xGrid=100,
            vGrid=50
    ):
        engine = set_fd_engine(self.process_type, self.process, tGrid, xGrid, vGrid)
        self.option.setPricingEngine(engine)
        return

    def impliedVolatility(self, NPV, stock_price=None):
        if stock_price is not None:
            # print('aaaa', stock_price)
            self.stock_price_quote.setValue(stock_price)
        try:
            return self.option.impliedVolatility(
            NPV, self.process
        )
        except:
            print('fail calculate impliedVolatility')

    def impliedVolatility_multi(self, NPVs, stock_prices):
        # np.vectorize(self.impliedVolatility)(NPVs, stock_prices, accuracy, maxEvaluations, minVol, maxVol)
        return [self.impliedVolatility(
            npv, price
        ) for npv, price in zip(NPVs, stock_prices)]

    def impliedVolatility_and_delta(
            self,
            NPV,
            pre_vol=np.nan,
            stock_price=None,
    ):
        if stock_price is not None:
            self.stock_price_quote.setValue(stock_price)
        try:
            vol = self.option.impliedVolatility(
                NPV, self.process
            )
            self.stock_vol_quote.setValue(vol)
            delta = self.option.delta()
        except:
            vol = pre_vol
            self.stock_vol_quote.setValue(vol)
            delta = self.option.delta()
            print(f'impliedVolatility_and_delta fail calculation: use pre volatility {vol}, delta {delta}')
        return vol, delta

    def impliedVolatility_and_delta_multi(
            self,
            NPVs,
            pre_vol=np.nan,  # 如果计算无法获得vol，则使用。
            stock_prices=None,
    ):
        vols, deltas = np.vectorize(self.impliedVolatility_and_delta)(NPVs, pre_vol, stock_prices)
        return vols, deltas

    def NPV(self):
        return self.option.NPV()

    def NPV_multi(self, prices: np.ndarray):
        return np.vectorize(self._cal_NPV)(prices)

    def delta(self):
        return self.option.delta()

    def delta_multi(self, prices: np.ndarray):
        return np.vectorize(self._cal_delta)(prices)

    def delta_numerical(self, h=0.01, stock_price=None, reset_today_quote=True):
        if reset_today_quote:
            pre_price = self.stock_price_quote.value()

        if stock_price is None:
            stock_price = self.stock_price_quote.value()

        price_up =  self._cal_NPV(stock_price + h)
        price_down = self._cal_NPV(stock_price - h)

        delta = (price_up - price_down) / (2 * h)

        if reset_today_quote:
            self.stock_price_quote.setValue(pre_price)

        return delta

    def delta_numerical_multi(self, prices:np.ndarray, h=0.01, reset_today_quote=True):
        if reset_today_quote:
            pre_price = self.stock_price_quote.value()

        price_up = np.vectorize(self._cal_NPV)(prices + h)
        price_down = np.vectorize(self._cal_NPV)(prices - h)

        delta = (price_up - price_down) / (2 * h)

        if reset_today_quote:
            self.stock_price_quote.setValue(pre_price)

        return delta

    def _cal_NPV(self, price):
        self.stock_price_quote.setValue(price)
        return self.option.NPV()

    def _cal_delta(self, price):
        self.stock_price_quote.setValue(price)
        return self.option.delta()






