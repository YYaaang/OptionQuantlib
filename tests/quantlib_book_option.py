#%% md
# https://www.quantlibguide.com/Instruments%20and%20pricing%20engines.html#other-pricing-methods
# 
# # Instruments and pricing engines
# This notebook showcases a couple of features that the infrastructure of the library makes available; namely, it will show how instruments can use different so-called pricing engines to calculate their prices (each engine implementing a given model and/or numerical method) and how engines and instruments can be notified of changes in their input data and react accordingly.
#%% md
# # Setup
# 
# To begin, we import the QuantLib module and set up the global evaluation date.
#%%
import QuantLib as ql

today = ql.Date(7, ql.March, 2024)
ql.Settings.instance().evaluationDate = today
#%% md
# # The instrument
# 
# In this notebook, we’ll leave fixed-income and take a textbook instrument example: a European option.
# 
# Building the option requires only the specification of its contract, so its payoff (it’s a call option with strike at 100) and its exercise, three months from today’s date. The instrument doesn’t take any market data; they will be selected and passed later, depending on the calculation method.
#%%
option = ql.EuropeanOption(
    ql.PlainVanillaPayoff(ql.Option.Call, 100.0),
    ql.EuropeanExercise(ql.Date(7, ql.June, 2024)),
)
#%% md
# # A first pricing method
# 
# The different pricing methods are implemented as pricing engines holding the required market data. The first we’ll use is the one encapsulating the analytic Black-Scholes formula.
# 
# First, we collect the quoted market data. We’ll assume flat risk-free rate and volatility, so they can be expressed by SimpleQuote instances: they model numbers whose value can change and that can notify observers when this happens. The underlying value is at 100, the risk-free value at 1%, and the volatility at 20%.
#%%
u = ql.SimpleQuote(100.0)
r = ql.SimpleQuote(0.01)
σ = ql.SimpleQuote(0.20)
#%% md
# In order to build the engine, the market data are encapsulated in a Black-Scholes process object. The process can use full-fledged term structures, so it can include time-dependency and smiles. In this case, for simplicity, we build flat curves for the risk-free rate and the volatility.
#%%
riskFreeCurve = ql.FlatForward(
    0, ql.TARGET(), ql.QuoteHandle(r), ql.Actual360()
)
volatility = ql.BlackConstantVol(
    0, ql.TARGET(), ql.QuoteHandle(σ), ql.Actual360()
)
#%% md
# Now we can instantiate the process with the underlying value and the curves we just built. The inputs are all stored into handles, so that we could change the quotes and curves used if we wanted. I’ll skip over this for the time being.
#%%
process = ql.BlackScholesProcess(
    ql.QuoteHandle(u),
    ql.YieldTermStructureHandle(riskFreeCurve),
    ql.BlackVolTermStructureHandle(volatility),
)
#%% md
# Once we have the process, we can finally use it to build the engine…
#%%
engine = ql.AnalyticEuropeanEngine(process)
#%% md
# …and once we have the engine, we can set it to the option and evaluate the latter.
#%%
option.setPricingEngine(engine)
#%%
print(option.NPV())
#%% md
# Depending on the instrument and the engine, we can also ask for other results; in this case, we can ask for Greeks.
#%%
print(option.delta())
print(option.gamma())
print(option.vega())
#%% md
# # Market changes
# 
# As I mentioned, market data are stored in Quote instances and thus can notify the option when any of them changes. We don’t have to do anything explicitly to tell the option to recalculate: once we set a new value to the underlying, we can simply ask the option for its NPV again and we’ll get the updated value.
#%%
u.setValue(105.0)
print(option.NPV())
#%% md
# Other market data also affect the value, of course.
#%%
r.setValue(0.02)
print(option.NPV())
#%%
σ.setValue(0.15)
print(option.NPV())
#%% md
# # Date changes
# 
# Just as it does when inputs are modified, the value also changes if we advance the evaluation date. Let’s look first at the value of the option when its underlying is worth 105 and there’s still three months to exercise…
#%%
u.setValue(105.0)
r.setValue(0.01)
σ.setValue(0.20)
print(option.NPV())
#%% md
# …and then move to a date two months before exercise.
#%%
ql.Settings.instance().evaluationDate = ql.Date(7, ql.April, 2024)
#%% md
# Again, we don’t have to do anything explicitly: we just ask the option for its value, and we see that it has decreased as expected.
#%%
print(option.NPV())
#%% md
# # A note on the option value on its exercise date
# In the default library configuration, the instrument is considered to have expired when it reaches the exercise date, so its returned value goes down to 0.
#%%
ql.Settings.instance().evaluationDate = ql.Date(7, ql.June, 2024)
#%%
print(option.NPV())
#%% md
# It’s possible to tweak the configuration so that the instrument is still considered alive.
#%%
ql.Settings.instance().includeReferenceDateEvents = True
#%% md
# The above changes the settings, but doesn’t send a notification to the instrument so we need to trigger an explicit recalculation. Normally, though, one would change the setting at the start of one’s program so this step would be unnecessary.
#%%
option.recalculate()

print(option.NPV())
#%% md
# However, this is not guaranteed to work for all pricing engines, since each one must manage this case specifically; and even when they return a price, they are not guaranteed to return meaningful values for all available results. For instance, at the time of this writing, the cell below will print two NaNs; if it doesn’t, please send me a line so I can update this text.
#%%
print(option.delta())
print(option.vega())
#%% md
# # Other pricing methods
# 
# As I mentioned, the instrument machinery allows us to use different pricing methods. For comparison, I’ll first set the input data back to what they were previously and output the Black-Scholes price.
#%%
ql.Settings.instance().evaluationDate = today
u.setValue(105.0)
r.setValue(0.01)
σ.setValue(0.20)
#%%
print(option.NPV())
#%% md
# Let’s say that we want to use a Heston model to price the option. What we have to do is to instantiate the corresponding class with the desired inputs (here I’ll skip the calibration and pass precalculated parameters)…
#%%
model = ql.HestonModel(
    ql.HestonProcess(
        ql.YieldTermStructureHandle(riskFreeCurve),
        ql.YieldTermStructureHandle(
            ql.FlatForward(0, ql.TARGET(), 0.0, ql.Actual360())
        ),
        ql.QuoteHandle(u),
        0.04,
        0.1,
        0.01,
        0.05,
        -0.75,
    )
)
#%% md
# …pass it to the corresponding engine, and set the new engine to the option.
#%%
engine = ql.AnalyticHestonEngine(model)
option.setPricingEngine(engine)
#%% md
# Asking the option for its NPV will now return the value according to the new model.
#%%
print(option.NPV())
#%% md
# # Lazy recalculation
# 
# One last thing. Up to now, we haven’t really seen evidence of notifications going around. After all, the instrument might just have recalculated its value every time we asked it, regardless of notifications. What I’m going to show, instead, is that the option doesn’t just recalculate every time anything changes; it also avoids recalculations when nothing has changed.
# 
# We’ll switch to a Monte Carlo engine, which takes a few seconds to run the required simulation.
#%%
engine = ql.MCEuropeanEngine(
    process, "PseudoRandom", timeSteps=20, requiredSamples=500_000
)
option.setPricingEngine(engine)
#%% md
# When we ask for the option value, we have to wait a noticeable time for the calculation to finish (for those of you reading this in a non-interactive way, I’ll also have the notebook output the time)…
#%%
%time print(option.NPV())
#%% md
# …but a second call to the NPV method will be instantaneous when made before anything changes. In this case, the option didn’t calculate its value; it just returned the result that it cached from the previous call.
#%%
%time print(option.NPV())
#%% md
# If we change anything (e.g., the underlying value)…
#%%
u.setValue(104.0)
#%% md
# …the option is notified of the change, and the next call to NPV will again take a while.
#%%
option
#%%
option.delta
#%%
