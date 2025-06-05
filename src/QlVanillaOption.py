import QuantLib as ql
from src.QlStockPricer import QlStockPricer
from typing import Optional, Union
import types

class QlVanillaOption(ql.VanillaOption):

    """
    增强版VanillaOption定价器，支持：
    1. 动态绑定定价引擎
    2. 自动计算希腊字母
    3. 参数校验与错误处理
    """
    def __init__(
        self,
        payoff: ql.PlainVanillaPayoff,
        exercise: ql.Exercise,
        QlStockPricer:QlStockPricer,
        engine: Optional[ql.PricingEngine] = None,
        **engine_params
    ):
        super().__init__(payoff, exercise)

        self.strike_price = payoff.strike()
        self.option_type = payoff.optionType()
        self.end_date = exercise.lastDate()

        self.StockPricer = QlStockPricer
        self.today = QlStockPricer.today
        self.stock_price = QlStockPricer.stock_price
        if engine is not None:
            self.set_engine(engine, **engine_params)

    @classmethod
    def init_from_price_and_date(
        cls,
        strike_price: float,
        end_date: ql.Date,
        QlStockPricer: QlStockPricer,
        option_type=ql.Option.Call,
        engine: Optional[ql.PricingEngine] = None,
        **engine_params
    ):
        payoff = ql.PlainVanillaPayoff(option_type, strike_price)
        exercise = ql.EuropeanExercise(end_date)
        return cls(payoff, exercise, QlStockPricer, engine, **engine_params)

    def set_engine(
        self,
        engine: Union[ql.PricingEngine, type],
        **params
    ) -> None:
        """动态设置定价引擎"""
        if isinstance(engine, types.FunctionType) or isinstance(engine, type):  # 传入引擎类而非实例
            process = self.StockPricer.get_process()
            if (type(process) == ql.HestonProcess) and (type(engine) in [ql.AnalyticHestonEngine, ql.FdHestonVanillaEngine]):
                hestonModel = ql.HestonModel(process)
                self._engine = engine(hestonModel, **params)
            else:
                self._engine = engine(process, **params)
        else:  # 直接传入引擎实例
            self._engine = engine
        self.setPricingEngine(self._engine)
        return

    def greeks(self) -> dict:
        """返回所有希腊字母"""
        return {
            'delta': self.delta(),
            'gamma': self.gamma(),
            'theta': self.theta(),
            'vega': self.vega(),
            'rho': self.rho()
        }

    # @classmethod
    # def create_european(
    #     cls,
    #     option_type: ql.Option.Type,
    #     strike: float,
    #     maturity_date: ql.Date,
    #     engine: Optional[ql.PricingEngine] = None,
    #     **engine_params
    # ) -> 'QlVanillaOptionPricer':
    #     """快速创建欧式期权"""
    #     payoff = ql.PlainVanillaPayoff(option_type, strike)
    #     exercise = ql.EuropeanExercise(maturity_date)
    #     return cls(payoff, exercise, engine, **engine_params)