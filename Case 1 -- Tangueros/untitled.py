import numpy as np
import scipy 
from scipy import stats
from scipy.stats import norm
import lmfit
import time


mkt_price = np.array([6.5757, 2.8223, 0.6335,
                      8.1165, 4.3850, 1.7263,
                      6.0865, 3.1820, 1.2317,
                      7.7710, 4.7369, 2.4165])

maturity_ = [0.25, 0.5, 0.75, 1]
strikes = np.array([[95, 100, 105], [95,100,105],[100,105,110],[100,105,110]])

# def resid(params, maturity, strikes, y_data):

#     eta_ = params['eta'].value
#     lam_ = params['lam'].value
#     rho_ = params['rho'].value
#     vmean_ = params['vmean'].value
#     model_data = list()
#     for m in range(len(maturity)):
#         for st in range(strikes.shape[1]):
#             price_ = Milstein(eta=eta_, lam=lam_, 
#                               rho=rho_, vmean=vmean_,
#                               K=strikes[m,st], T=maturity[m])
#             model_data.append(price_)
#     return np.square(np.array(price_) - np.array(y_data)).sum()

def bs_price(sigma, S, T, K, r=0.0411):
    K = K.reshape(12)
    T = T.reshape(12)
    d1 = (np.log(S/K) + (r + sigma ** 2 / 2) * T) / sigma / np.sqrt(T)
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
def resid(params, S0, maturity, strike, y_data):

    sigma = params['sigma'].value
    bs_prices = bs_price(sigma, S0, maturity, strike)

    return np.square(np.array(bs_prices) - np.array(y_data)).sum()





a = time.time()
params = lmfit.Parameters()
params.add('eta', 1.0, min=0, max=2)
params.add('lam', 1.0, min=0, max=2)
params.add('rho', 0.5, min=-0.8, max=0.8)
params.add('vmean', 0.2, min=0.01, max=0.5)

o1 = lmfit.minimize(resid, params, args=(maturity_, strikes, mkt_price), method='brute',Ns=10, workers=-1)
print("# Fit using differential_evolution:")
lmfit.report_fit(o1)
b = time.time()
print("Time Elapsed:", b-a)


# print(Milstein())
