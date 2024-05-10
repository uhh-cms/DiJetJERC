# coding: utf-8

def linear_function(x, p):
    """Linear function in `x` with slope `p[0]` and intercept `p[1]`."""
    return p[0] * x + p[1]


def chi2_linear(p, x, data, cov_inv):
    """
    Chi2 function to minimize assuming the model follows a linear function
    of `x` with parameters `p`, compared to `data` with inverse covariance
    matrix `cov_inv`.
    """
    y_hat = linear_function(x=x, p=p)
    residuals = data - y_hat
    chi2 = residuals.T @ cov_inv @ residuals
    return chi2
