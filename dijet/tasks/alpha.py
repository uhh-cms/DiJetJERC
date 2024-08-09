# coding: utf-8

"""
Custom tasks to derive JER SF.
"""

import law

from columnflow.tasks.framework.base import Requirements
from columnflow.util import maybe_import

from dijet.tasks.asymmetry import Asymmetry
from dijet.tasks.base import HistogramsBaseTask
from dijet.tasks.correlated_fit import CorrelatedFit

hist = maybe_import("hist")
np = maybe_import("numpy")
it = maybe_import("itertools")


class AlphaExtrapolation(
    HistogramsBaseTask,
    CorrelatedFit,
):
    """
    Task to perform alpha extrapolation.
    Read in and plot asymmetry histograms.
    Extrapolate sigma_A( alpha->0 ).
    """

    # Add nested sibling directories to output path
    output_collection_cls = law.NestedSiblingFileCollection

    # upstream requirements
    reqs = Requirements(
        Asymmetry=Asymmetry,
    )

    def requires(self):
        return self.reqs.Asymmetry.req(self)

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["key"] = self.as_branch().requires()
        return reqs

    def load_asymmetries(self):
        histogram = self.input()["asym"].load(formatter="pickle")
        return histogram

    def load_integrals(self):
        histogram = self.input()["nevt"].load(formatter="pickle")
        return histogram

    def output(self) -> dict[law.FileSystemTarget]:
        # TODO: Unstable for changes like data_jetmet_X
        #       Make independent like in config datasetname groups
        sample = self.extract_sample()
        target = self.target(f"{sample}", dir=True)
        # declare the main target
        outp = {
            "widths": target.child("widths.pickle", type="f"),
            "extrapolation": target.child("extrapolation.pickle", type="f"),
        }
        return outp

    def run(self):
        # TODO: Gen level for MC
        #       Correlated fit (in jupyter)

        h_asyms = self.load_asymmetries()
        h_nevts = self.load_integrals()

        # Get widths of asymmetries
        asyms = h_asyms.axes[self.asymmetry_variable].centers

        # Take mean value from normalized asymmetry
        axes_names = [a.name for a in h_asyms.axes]
        assert axes_names[-1] == self.asymmetry_variable, "asymmetry axis must come last"
        means = np.nansum(
            asyms * h_asyms.view().value,
            axis=-1,
            keepdims=True,
        )
        h_stds = h_asyms.copy()
        h_stds = h_stds[{self.asymmetry_variable: sum}]

        # Get stds
        h_stds.view().value = np.sqrt(
            np.average(
                ((asyms - means)**2),
                weights=h_asyms.view().value,
                axis=-1,
            ),
        )
        h_stds.view().value = np.nan_to_num(h_stds.view().value, nan=0.0)

        # note: error on std deviation analogous to implementation in ROOT::TH1
        # https://root.cern/doc/v630/TH1_8cxx_source.html#l07520
        h_stds.view().variance = h_stds.values()**2 / (2 * h_nevts.values())
        h_stds.view().variance = np.nan_to_num(h_stds.view().variance, nan=0.0)

        # Store alphas here to get alpha up to 1
        # In the next stepts only alpha<0.3 needed; avoid slicing from there
        results_widths = {
            "widths": h_stds,
        }
        self.output()["widths"].dump(results_widths, formatter="pickle")

        # Get max alpha for fit; usually 0.3
        amax = 0.3  # TODO: define in config
        h_stds = h_stds[{self.alpha_variable: slice(0, hist.loc(amax))}]
        h_nevts = h_nevts[{self.alpha_variable: slice(0, hist.loc(amax))}]
        # exclude 0, the first bin, from alpha edges
        alphas = h_stds.axes[self.alpha_variable].edges[1:]

        # TODO: More efficient procedure than for loop?
        #       - Idea: Array with same shape but with tuple (width, error) as entry
        n_bins = [
            len(h_stds.axes[bv].centers)
            for bv in self.binning_variables
        ]
        n_methods = len(h_stds.axes["category"].centers)  # ony length
        inter = h_stds.copy().values()
        inter = inter[:, :2, ...]  # keep first two entries
        slope = h_stds.copy().values()
        slope = slope[:, :2, ...]  # keep first two entries
        for m, *bv_indices, p in it.product(
            range(n_methods),
            *[range(n) for n in n_bins]
        ):
            h_slice = {
                "category": m,
            }
            h_slice.update({
                bv: bv_index
                for bv, bv_index in zip(
                    self.binning_variables,
                    bv_indices,
                )
            })
            tmp = h_stds[h_slice]
            tmp_evts = h_nevts[h_slice]
            coeff, err = self.get_correlated_fit(wmax=alphas, std=tmp.values(), nevts=tmp_evts.values())
            inter[m, :, *bv_indices] = [coeff[1], err[1]]
            slope[m, :, *bv_indices] = [coeff[0], err[0]]

        # NOTE: store fits into hist.
        h_intercepts = h_stds.copy()
        # Remove axis for alpha for histogram
        h_intercepts = h_intercepts[{self.alpha_variable: sum}]
        # y intercept of fit (x=0)
        h_intercepts.view().value = inter[:, 0, ...]
        # Errors temporarly used; Later get:
        # Error on fit from fit function (how?) or new method with three fits
        h_intercepts.view().variance = inter[:, 1, ...]

        h_slopes = h_intercepts.copy()
        # Slope of fit stored in index 1
        h_slopes.view().value = slope[:, 0, ...]
        # Only stored for plotting, no defined error
        h_slopes.view().variance = slope[:, 1, ...]

        results_extrapolation = {
            "intercepts": h_intercepts,
            "slopes": h_slopes,
        }
        self.output()["extrapolation"].dump(results_extrapolation, formatter="pickle")
