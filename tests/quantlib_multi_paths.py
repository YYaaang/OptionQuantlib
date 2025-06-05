import QuantLib as ql
import numpy as np
import time

timestep, length, numPaths = 1, 2, 5

today = ql.Date().todaysDate()
riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.05, ql.Actual365Fixed()))
dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.01, ql.Actual365Fixed()))
initialValue = ql.QuoteHandle(ql.SimpleQuote(100))

v0, kappa, theta, rho, sigma = 0.005, 0.8, 0.008, 0.2, 0.1
hestonProcess = ql.HestonProcess(riskFreeTS, dividendTS, initialValue, v0, kappa, theta, sigma, rho)

times = ql.TimeGrid(length, timestep)
dimension = hestonProcess.factors()

rng = ql.UniformLowDiscrepancySequenceGenerator(dimension * timestep)
sequenceGenerator = ql.GaussianLowDiscrepancySequenceGenerator(rng)
pathGenerator = ql.GaussianSobolMultiPathGenerator(hestonProcess, list(times), sequenceGenerator, False)

t = time.time()
a = [np.array(pathGenerator.next().value()) for i in range(numPaths)]
print(time.time() - t)

t = time.time()
aa = [pathGenerator.next().value() for i in range(numPaths)]
aaa = [aa[i][0][1] for i in range(numPaths)]
print('2      ', time.time() - t)

t = time.time()
# paths[0] will contain spot paths, paths[1] will contain vol paths
paths = [[] for i in range(dimension)]
for i in range(numPaths):
    samplePath = pathGenerator.next()
    values = samplePath.value()
    spot = values[0]

    for j in range(dimension):
        paths[j].append([x for x in values[j]])
print(time.time() - t)
print()