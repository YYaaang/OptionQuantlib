import pandas as pd
import numpy as np
import QuantLib as ql
import time
from config.config import RANDOM_SEED
from typing import Union, List

from src.utils import check_list_length

def black_scholes_merton_model(
        ql_calendar,
        price_quotes: Union[ql.SimpleQuote, List[ql.SimpleQuote]] = 100,
        vol_quotes: Union[ql.SimpleQuote, List[ql.SimpleQuote]] = 0.2,
        dividend_rate_quotes: Union[ql.SimpleQuote, List[ql.SimpleQuote]] = 0.0,
        settlementDays=0,
        is_merton=False,
):

    volatility_handle = [
        ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(
                settlementDays,
                ql_calendar.calendar,
                ql.QuoteHandle(s),
                ql_calendar.day_counter)
        ) for s in vol_quotes
    ]

    if is_merton:
        # process_type = 'black_schole_merton'
        dividend_curve_handle = [
            ql.YieldTermStructureHandle(
                ql.FlatForward(
                    settlementDays,  # settlementDays
                    ql_calendar.calendar,
                    ql.QuoteHandle(rate),
                    ql_calendar.day_counter
                )
            ) for rate in dividend_rate_quotes
        ]

        process = [
            ql.BlackScholesMertonProcess(
                ql.QuoteHandle(price),
                dividend,
                ql_calendar.risk_free_rate_curve_handle,
                vol
            ) for price, dividend, vol in zip(price_quotes, dividend_curve_handle, volatility_handle)
        ]

    else:
        # process_type = 'black_schole'

        process = [
            ql.BlackScholesProcess(
                ql.QuoteHandle(price),
                ql_calendar.risk_free_rate_curve_handle,
                vol
            ) for price, vol in zip(price_quotes, volatility_handle)
        ]

    return process

def heston_model(
        ql_calendar,
        price_quotes: Union[ql.SimpleQuote, List[ql.SimpleQuote]]=100,
        v0: Union[float, List[float]] = 0.005,
        kappa: Union[float, List[float]] = 0.8,
        theta: Union[float, List[float]] = 0.008,
        rho: Union[float, List[float]] = 0.2,
        sigma: Union[float, List[float]] = 0.1,
        dividend_rate_quotes: Union[ql.SimpleQuote, List[ql.SimpleQuote]] = 0.0,
        settlementDays=0
):

    dividend_curve_handle = [
        ql.YieldTermStructureHandle(
            ql.FlatForward(
                settlementDays,  # settlementDays
                ql_calendar.calendar,
                ql.QuoteHandle(rate),
                ql_calendar.day_counter
            )
        ) for rate in dividend_rate_quotes
    ]

    processes = [
        ql.HestonProcess(
            ql_calendar.risk_free_rate_curve_handle,
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

    return processes
