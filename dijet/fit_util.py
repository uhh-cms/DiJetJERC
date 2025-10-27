# coding: utf-8

from columnflow.util import maybe_import

sc = maybe_import("scipy.optimize")
nd = maybe_import("numdifftools")
np = maybe_import("numpy")


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


class CorrelatedFit():
    """
    Tool for fitting a linear function to widths as a function of inclusive
    alpha bins, taking correlations into account as appropriate.

    ## Covariance matrix
    [1] Based on K. Goebel dissertation ch. A.3.
    https://www.physik.uni-hamburg.de/en/iexp/gruppe-haller/scientific-output/documents/thesis-kristin-goebel.pdf

    Example code here
    https://github.com/UHH2/DiJetJERC/blob/ff98eebbd44931beb016c36327ab174fdf11a83f/
    JERSF_Analysis/JER/wide_eta_binning/functions.C#L558
    func. createCov()
    """

    @staticmethod
    def create_cov(widths, nevts):
        """
        Calculate a covariance matrix for a sequence of distribution widths
        correspondin to inclusive alpha bins. The input values must be in
        order of ascending alpha.

        widths: array of widths of the asymmetry distributions in the inclusive alpha bins
        nevts: array of event counts in the inclusive alpha bins
        """
        # square of the uncertainty on the width from normal approximation
        # note: error on std deviation analogous to implementation in ROOT::TH1
        # https://root.cern/doc/v630/TH1_8cxx_source.html#l07520
        widths_err2 = widths**2 / (2 * nevts)

        # NxN matrices mapping pair of values (a_i, a_j) to that
        # corresponding to the lower (higher) index,
        # i.e. entry at (i, j) = min(i, j) or max(i, j)
        idx = np.arange(len(widths))
        min_idx_matrix = np.minimum.outer(idx, idx)
        max_idx_matrix = np.maximum.outer(idx, idx)

        # obtain ratios of widths and event counts
        # for larger over smaller index
        width_matrix = widths[min_idx_matrix] / widths[max_idx_matrix]
        n_matrix = nevts[min_idx_matrix] / nevts[max_idx_matrix]

        # calculate covariance and return
        cov_mat = width_matrix * n_matrix * widths_err2
        return np.nan_to_num(cov_mat)

    @staticmethod
    def correlated_fit(wmax, data, cov_inv):
        # Set the initial parameters.
        params = np.array([0.05, 0.15])

        # Minimize the correlated fit error.
        result = sc.minimize(
            chi2_linear,
            params,
            args=(wmax, data, cov_inv),
        )

        hess = nd.Hessian(chi2_linear)(result.x, wmax, data, cov_inv)
        hess_inv = np.linalg.inv(hess)

        pcov = 2.0 * hess_inv
        perr = np.sqrt(np.diag(pcov))

        return result.x, perr

    @staticmethod
    def get_correlated_fit(wmax, std, nevts):
        # In case of very few events in a eta-pt bin two alpha bins can be equal
        # In that case the matrix is not invertable "numpy.linalg.LinAlgError: Singular matrix"
        # np.insert(std[:-1] != std[1:], 0, True) sets entry i to False if entry i-1 is equal
        mask = (
            (std != 0) &
            np.insert(std[:-1] != std[1:], 0, True)
        )
        if len(mask[mask]) < 2:
            return [[0, 0], [0, 0]]

        wmax = wmax[mask]
        std = std[mask]
        nevts = nevts[mask]

        y_cov_mc = CorrelatedFit.create_cov(widths=std, nevts=nevts)
        y_cov_mc_inv = np.linalg.inv(y_cov_mc)
        popt, perr = CorrelatedFit.correlated_fit(wmax, std, y_cov_mc_inv)

        return popt, perr
