#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 22:16:36 2019

@author: yuxian
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.linear_model import LinearRegression
import math
from scipy.optimize import minimize
import pandas as pd
import lmfit
import pdb

def L0(x):
    return (np.exp(-x/2)).reshape(-1, 1)
def L1(x):
    return (np.exp(-x/2)*(1-x)).reshape(-1, 1)
def L2(x):
    return (np.exp(-x/2)*(1-2*x+x**2/2)).reshape(-1, 1)


state = np.zeros((20000, 96))
state[(np.arange(20000) % 2) == 1, :] = 1
state[:, 0] = 0

power = pd.read_csv("Power_Simulation_97.csv", header=None)
energy = pd.read_csv("Gas_Simulation_97.csv", header=None)
spark_spread = (power - 12*energy).T.values
spark_spread = np.array([spark_spread[i//2,:] for i in range(20000)])



r = 0.05
start_cost = 3 * 1000 * 16 * 7.6
discount = np.exp(-r/48)

def R(ss_t):
    return ss_t * 1000 * 16 * 7.6 - 5 * 1000 * 16 * 7.6


def get_xlist(x):
    return np.concatenate((L0(x), L1(x), L2(x)), axis=1)

def J(para, s0, ss0, ss1, J_delay, opt):
#     return profit of this week + revenue in the future
#    pdb.set_trace()
    if opt:
        upper = para['upper'].value
        lower = para['lower'].value
    else:
        upper, lower = para

    idx_off = s0 == 0
    idx_on = s0 == 1
#    pdb.set_trace()
    turn_on = idx_off * (ss0 > upper)
    turn_off = idx_on * (ss0 < lower)

    profit = R(ss1)
    profit[turn_on] = - start_cost
    profit[idx_off & (ss0 <= upper)] = 0

    a_t = idx_off * (ss0 > upper) + idx_on * (ss0 >= lower)

    X = get_xlist(ss0)
    Y = discount*(profit + J_delay)
    Y_hat = np.zeros(len(s0))
    
#    Y_hat[s0 == 0] = np.maximum(LinearRegression().fit(X[s0 == 0, :],Y[a_t == 1]).predict(X[s0 == 0, :]), LinearRegression().fit(X[s0==0, :], Y[a_t == 0].predict(X[s0 == 0, :])))
#    Y_hat[s0 == 1] = np.maximum(LinearRegression().fit(X[s0 == 1, :],Y[a_t == 1]).predict(X[s0 == 1, :]), LinearRegression().fit(X[s0==1, :], Y[a_t == 0].predict(X[s0 == 1, :])))

    
    if all(s0 == 0):   
#        if any(a_t == 0):
#            idx = (s0==0) & (a_t==0)
#            Y_hat[idx] = LinearRegression().fit(X[idx,:],Y).predict(X[idx,:])
#        if any(a_t == 1):
#            idx = (s0==0) & (a_t==1)
#            Y_hat[idx] = LinearRegression().fit(X[idx,:],Y).predict(X[idx,:])
        Y_hat = Y[np.arange(len(Y)) + a_t]
        
        
    else:    

        if any((s0==0) & (a_t==0)):
            idx = (s0==0) & (a_t==0)
            Y_hat[idx] = LinearRegression().fit(X[idx,:],Y[np.arange(0, len(s0), 2)][a_t[s0==0]==0]).predict(X[idx,:])
        if any((s0==0) & (a_t==1)):
            idx = (s0==0) & (a_t==1)
            Y_hat[idx] = LinearRegression().fit(X[idx,:],Y[np.arange(1, len(s0), 2)][a_t[s0==0]==1]).predict(X[idx,:])

        if any((s0==1) & (a_t==0)):
            idx = (s0==1) & (a_t==0)
            Y_hat[idx] = LinearRegression().fit(X[idx,:],Y[np.arange(0, len(s0), 2)][a_t[s0==1]==0]).predict(X[idx,:])
        if any((s0==1) & (a_t==1)):
            idx = (s0==1) & (a_t==1)
            Y_hat[idx] = LinearRegression().fit(X[idx,:],Y[np.arange(1, len(s0), 2)][a_t[s0==1]==1]).predict(X[idx,:])
#    
#    if not opt:
#        pdb.set_trace()
    return Y_hat

def target(para, s0, ss0, ss1, J_delay, opt):
    return -np.mean(J(para, s0, ss0, ss1, J_delay, opt))

def optimize(state, spark, J_delay=None):
    thres = np.zeros((2, spark.shape[1] - 1))
    value = np.zeros((state.shape))

    for i in range(state.shape[1])[::-1]:
        if J_delay is None:
            J_delay = np.zeros(state.shape[0])
            
        s0 = state[:, i]    
        ss0 = spark[:, i]
        ss1 = spark[:, i+1]
        params = lmfit.Parameters()
        params.add('upper', 5, min=-5, max=15)
        params.add('lower', 0, min=-10, max=10)
        bound_result = lmfit.minimize(target, params, args=(s0, ss0, ss1, J_delay, True), method="brute", Ns=10, workers=-1) 

        thres[:, i] = [bound_result.params['upper'].value, bound_result.params['lower'].value]
        J_delay = J(thres[:, i], s0, ss0, ss1, J_delay, False)
#        pdb.set_trace()
        value[:,i] = J_delay
        
    return thres, value


thres, value = optimize(state, spark_spread)