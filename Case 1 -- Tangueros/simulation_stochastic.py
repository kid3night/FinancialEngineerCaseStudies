import numpy as np
import scipy 
from scipy import stats
from scipy.stats import norm
import lmfit
import time

def Milstein(
             eta = 1, 
             lam = 1, 
             rho = 0,
             vmean = 0.1,
             S0 = 100,   #given
             K = 100,    #given
             r = 0.0411, #given
             v0 = 0.11, # given
             N = 500,
             n = 10000,
             T = 1):

    def payoff(S,K):
        return np.max(S-K,0)
    
    # N: number of intervals
    # n: number of paths
    # T: maturity
    dt = float(T/N) #interval of each path
   
    W = norm.rvs(size = (n,N))*np.sqrt(dt)
    Wtilde = norm.rvs(size = (n,N))*np.sqrt(dt)
    B = rho*W + np.sqrt(1-rho**2)*Wtilde
    
    V = np.zeros((n,N+1)) # V is our variance 
    V[:,0] = v0
    S = np.zeros((n,N+1)) # S is our stock price
    S[:,0] = S0
    
    
    for i in range(1,N+1):
        V[:,i] = V[:,i-1] - lam*(V[:,i-1] -vmean)*dt + eta*np.sqrt(V[:,i-1])*B[:,i-1]+\
        0.25*eta**2*(B[:,i-1]**2 - dt)
        V[:,i] = np.abs(V[:,i])
        
    dlogS = (r - 0.5*V[:,:-1])*dt + np.sqrt(V[:,:-1])*W
    dlogS_sum = np.cumsum(dlogS,axis = 1)
    S[:,1:] = S0*np.exp(dlogS_sum)
    price = np.mean(np.exp(-r*T)*payoff(S[:,N],K))
    return price

mkt_price = np.array([6.5757, 2.8223, 0.6335,
                      8.1165, 4.3850, 1.7263,
                      6.0865, 3.1820, 1.2317,
                      7.7710, 4.7369, 2.4165])
maturity_ = [0.25, 0.5, 0.75, 1]
strikes = np.array([[95, 100, 105], [95,100,105],[100,105,110],[100,105,110]])

def resid(params, maturity, strikes, y_data):

    eta_ = params['eta'].value
    lam_ = params['lam'].value
    rho_ = params['rho'].value
    vmean_ = params['vmean'].value
    model_data = list()
    for m in range(len(maturity)):
        for st in range(strikes.shape[1]):
            price_ = Milstein(eta=eta_, lam=lam_, 
                              rho=rho_, vmean=vmean_,
                              K=strikes[m,st], T=maturity[m])
            model_data.append(price_)
    return np.square(np.array(price_) - np.array(y_data)).sum()

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
