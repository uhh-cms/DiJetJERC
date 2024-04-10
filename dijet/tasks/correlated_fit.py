# coding: utf-8

import law
from columnflow.util import maybe_import

from dijet.tasks.util import chi2

sc = maybe_import("scipy.optimize")
nd = maybe_import("numdifftools")
np = maybe_import("numpy")

class CorrelatedFit():
    """
    Task to calculated correlated fit to widths.

    ## Covariance matrix 
    [1] Based on K. Goebel dissertation ch. A.3.
    https://www.physik.uni-hamburg.de/en/iexp/gruppe-haller/scientific-output/documents/thesis-kristin-goebel.pdf

    Example code here
    https://github.com/UHH2/DiJetJERC/blob/ff98eebbd44931beb016c36327ab174fdf11a83f/
    JERSF_Analysis/JER/wide_eta_binning/functions.C#L558
    func. createCov()
    """

    def createCov(self, widths, widths_err):
        widths_err2 = pow(widths_err, 2)  # square each element

        # 3x3 matrix with the smaller error of indices i&j (Check [1])
        # [1,2,3] ->
        # [
        #  [1,1,1]
        #  [1,2,2]
        #  [1,2,3]
        # ]
        matrix_err = np.maximum.outer(widths_err2,widths_err2)

        ratio = ( pow(widths,2) / (2 * widths_err2) )
        n_ratio = pow((ratio[:, None] / ratio),2)  # get ij element
        w_ratio = (widths[:, None] / widths)  # get ij element

        # Calculate the upper triangle of the covariance matrix.
        y_cov_mc = np.triu(matrix_err * w_ratio * n_ratio)

        # Fill in the lower triangle of the covariance matrix.
        y_cov_mc += y_cov_mc.T - np.diag(y_cov_mc.diagonal())

        return np.nan_to_num(y_cov_mc)

    def correlated_fit(self, wmax, data, cov):
        # Set the initial parameters.
        params = np.array([0.05, 0.15])
        bounds = [(0.001,0.15), (0.05,0.5)]

        # Minimize the correlated fit error.
        result = sc.minimize(
            chi2,
            params,
            args=(wmax, data, cov)
        )

        hess = nd.Hessian(chi2)(result.x, wmax, data, cov)
        hess_inv = np.linalg.inv(hess)

        pcov = 2.0 * hess_inv
        perr = np.sqrt(np.diag(pcov))

        return result.x, perr

    def get_correlated_fit(self, wmax, std, err):
        # In case of very few events in a eta-pt bin two alpha bins can be equal
        # In that case the matrix is not invertable "numpy.linalg.LinAlgError: Singular matrix"
        # np.insert(std[:-1] != std[1:], 0, True) sets entry i to False if entry i-1 is equal
        mask = (
            (std != 0) &
            np.insert(std[:-1] != std[1:], 0, True)
        )
        if len(mask[mask]) < 2:
                return [[0,0], [0,0]]

        wmax = wmax[mask]
        std = std[mask]
        err = err[mask]

        y_cov_mc = self.createCov(widths=std, widths_err=err)
        y_cov_mc_inv = np.linalg.inv(y_cov_mc)
        popt, perr = self.correlated_fit(wmax, std, y_cov_mc_inv)

        return popt, perr
