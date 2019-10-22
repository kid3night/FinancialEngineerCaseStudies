import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


mkt_price = np.array([[6.5757, 2.8223, 0.6335], [8.1165, 4.3850, 1.7263],
                      [6.0865, 3.1820, 1.2317], [7.7710, 4.7369, 2.4165]])

def bs_price(sigma, S, T, K, r=0.0411):
    K = K.reshape(12)
    T = T.reshape(12)
    d1 = (np.log(S/K) + (r + sigma ** 2 / 2) * T) / sigma / np.sqrt(T)
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
def loss_fun(sigma, S, T, K, r=0.0411, mkt = mkt_price.reshape(12)):
    return np.sum((bs_price(sigma, S, T, K, r) - mkt) ** 2)
    
    
def imp_vol(S, T, K, r=0.0411):
    result = minimize(loss_fun, np.zeros(12)+0.1, (S, T, K), method='BFGS')
    return result

T = np.repeat(np.array([[0.25], [0.5], [0.75], [1]]), 3, axis=1)
K = np.array([np.arange(95, 110, 5), np.arange(95, 110, 5), np.arange(100, 115, 5), np.arange(100, 115, 5)])
implied_vol = imp_vol(100, T, K)
vol_surface = implied_vol.x.reshape(4,3)
print(vol_surface)
print(vol_surface[0,2] - vol_surface[0,0])
print(vol_surface[3,2] - vol_surface[3,0])
