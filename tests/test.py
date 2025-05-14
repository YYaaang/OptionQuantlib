#%%
import QuantLib as ql
import pandas as pd

#%%
# Begin by setting the valuation date of which the cap and the floor should be priced at
ql.Settings.instance().evaluationDate = ql.Date(1, 1, 2022)
# Then we initialize the curve we want to use for discounting and forecasting
discount_factors = [1, 0.965, 0.94]  # discount factors
dates = [
    ql.Date(1, 1, 2022),
    ql.Date(1, 1, 2023),
    ql.Date(1, 1, 2024),
]  # maturity dates of the discount factors
day_counter = ql.Actual360()
# Note that we will not strip a curve here, but simply use the discount factors and the dates defined above
# By default QuantLib DiscountCurve will log linearly interpolate between the points.
discount_curve = ql.DiscountCurve(dates, discount_factors, day_counter)
# The curve will note be linked in case we want to update the quotes later on
discount_handle = ql.YieldTermStructureHandle(discount_curve)

#%%


