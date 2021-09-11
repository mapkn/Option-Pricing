# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 22:46:46 2016

@author: Mitul
"""


def bsm_call_value(S0, K, T, r, sigma):
    """ Valuation of European call option in BSM model.
    Analytical formula.
    Parameters
    ==========
    S0 : float
        initial stock/index level
    K : float
        strike price
    T : float
        maturity date (in year fractions)
    r : float
        constant risk-free short rate
    sigma : float
        volatility factor in diffusion term
    Returns
    =======
    value : float
        present value of the European call option
    """
    from math import log,sqrt,exp
    from scipy import stats
    import numpy as np
    
    #S0=float(S0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    value = (S0 * stats.norm.cdf(d1, 0.0, 1.0)- K * exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))
  
    return value
    

def bsm_put_value(S0,K,T,r,sigma):
    """ Valuation of European call option in BSM model.
    Analytical formula.
    Parameters
    ==========
    S0 : float
        initial stock/index level
    K : float
        strike price
    T : float
        maturity date (in year fractions)
    r : float
        constant risk-free short rate
    sigma : float
        volatility factor in diffusion term
    Returns
    =======
    value : float
        present value of the European call option
    """
    from math import log,sqrt,exp
    from scipy import stats
    import numpy as np
    
    #S0=float(S0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    value = K * exp(-r * T) * stats.norm.cdf(-d2, 0.0, 1.0)-S0 * stats.norm.cdf(-d1, 0.0, 1.0)
  
    return value
    
    
def bsm_call_vega(S0, K, T, r, sigma):
    """ Vega of European option in BSM model.
    Parameters
    ==========
    S0 : float initial stock/index level
    K : float strike price
    T : float maturity date (in year fractions)
    r : float constant risk-free short rate
    sigma : float volatility factor in diffusion term
    Returns
    =======
    vega : float
    partial derivative of BSM formula with respect
    to sigma, i.e. Vega"""

    from math import log, sqrt
    from scipy import stats
    import numpy as np
    
    #S0 = float(S0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    vega = S0 * stats.norm.pdf(d1, 0.0, 1.0) * sqrt(T)
    return vega


def bsm_put_vega(S0, K, T, r, sigma):
    """ Vega of European option in BSM model.
    Parameters
    ==========
    S0 : float
    initial stock/index level
    K : float
    strike price
    T : float
    maturity date (in year fractions)
    r : float
    constant risk-free short rate
    sigma : float
    volatility factor in diffusion term
    Returns
    =======
    vega : float
    partial derivative of BSM formula with respect
    to sigma, i.e. Vega"""

    from math import log, sqrt
    from scipy import stats
    import numpy as np

    # S0 = float(S0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    vega = S0 * stats.norm.pdf(d1, 0.0, 1.0) * sqrt(T)
    return vega


def bsm_call_delta(S0, K, T, r, sigma):
    from math import log, sqrt
    from scipy import stats
    import numpy as np
    
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))

    delta = stats.norm.cdf(d1)
    return delta


def bsm_call_gamma(S0, K, T, r, sigma):
    """ Gamma of European option in BSM model.
    Parameters
    ==========
    S0 : float
    initial stock/index level
    K : float
    strike price
    T : float
    maturity date (in year fractions)
    r : float
    constant risk-free short rate
    sigma : float
    volatility factor in diffusion term
    Returns
    =======
    gamma : float
    second partial derivative of BSM formula with respect
    to S, i.e. Gamma"""
    
    from math import log, sqrt
    from scipy import stats
    import numpy as np
    
    #S0 = float(S0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    gamma = stats.norm.pdf(d1, 0.0, 1.0)/ (S0 * sigma * sqrt(T))
    return gamma

    
def bsm_put_delta(S0, K, T, r, sigma):
    from math import log, sqrt
    from scipy import stats
    import numpy as np
    
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    
    delta=stats.norm.cdf(d1)-1
    
    return delta
    
# Implied volatility function
def bsm_call_imp_vol(S0, K, T, r, C0, sigma_est, it):
    """ Implied volatility of European call option in BSM model.
    Parameters
    ==========
    S0 : float
    initial stock/index level
    K : float
    strike price
    T : float
    maturity date (in year fractions)
    r : float
    constant risk-free short rate
    sigma_est : float
    estimate of impl. volatility
    it : integer
    number of iterations
    Returns
    =======
    simga_est : float
    numerically estimated implied volatility"""
    for i in range(it):
        sigma_est -= ((bsm_call_value(S0, K, T, r, sigma_est) - C0)
        / bsm_vega(S0, K, T, r, sigma_est))
    return sigma_est    

def fwd_price(S_0,r,q,T):
    """ Forward price for an investment asset providing continuous dividend 
    yield rate q

    S0 : float
    initial stock/index level
    
    r : float
    constant risk-free short rate
    
    
    """
    from math import exp
    F_0=S_0*exp((r-q)*T)
    return F_0
    
def fwd(F_0,K,r,T):
    """Value of a forward contract 
    Parameters
    ==========
    F_0: float
    current forward price of the asset
    K : float
    delivery price of forward contract
    T : float
    maturity date (in year fractions)
    r : float
    constant risk-free short rate
    
    """
    from math import exp
    f=(F_0-K)*exp(-r*T)
    return f
    
