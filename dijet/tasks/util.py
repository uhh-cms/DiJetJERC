# coding: utf-8

def linear_function(x, p):
    return p[0] * x + p[1]

def chi2(p, x, data, cov):
    y_hat = linear_function(x=x, p=p)
    residuals = data - y_hat
    chi2 = residuals.T @ cov @ residuals
    return chi2