from scipy import integrate
import numpy as np

def beta(z, lam, rho, eta):
    return lam - rho*eta*z

def alpha(z):
    return z ** 2 / 2 - z / 2

def d(z, lam, rho, eta):
    return cmath.sqrt(beta(z, lam, rho, eta) ** 2 - 2 * alpha(z) * eta ** 2)

def r(z, lam, rho, eta, sign):
    return (beta(z, lam, rho, eta) + sign * d(z, lam, rho, eta)) / eta ** 2

def g(z, lam, rho, eta):
    return r(z, lam, rho, eta, -1) / r(z, lam, rho, eta, 1)

def C(z, lam, rho, eta, T):
    return lam * r(z, lam, rho, eta, -1) * T - \
2 * lam / eta ** 2 * cmath.log((1 - g(z, lam, rho, eta) * cmath.exp(-d(z, lam,rho, eta)*T)) / (1 - g(z, lam, rho, eta)))

def D(z, lam, rho, eta, T):
    return r(z, lam, rho, eta, -1) * ((1 - cmath.exp(-d(z, lam, rho, eta) * T)) / (1 - g(z, lam, rho, eta) * cmath.exp(-d(z, lam,rho, eta)*T)))

def fourier(img, real, eta, lam, rho, vmean, K, T, S0=100, v0=0.11):
    z = real + 1j * img
    eq =  cmath.exp(np.log(S0) * z) * cmath.exp(C(z, lam, rho, eta, T) * vmean + D(z, lam, rho, eta, T) * v0) 
    hhat = cmath.exp(np.log(K) * (1 - z)) / (z * (1 - z))
    return hhat * eq


def intergral_numerical2(func, func2, x1, x2):
    myfun2_re = integrate.quad(func, x1, x2)[0]
    myfun2_im = integrate.quad(func2, x1, x2)[0]
    myfun2 = myfun2_re + 1j*myfun2_im
    return myfun2



def f1_(x):
    return np.real(np.exp(-x))

def f2_(x):
    return np.imag(np.exp(-x))

if __name__ == '__main__':
    print(intergral_numerical2(f1_, f2_, 1, np.inf))

