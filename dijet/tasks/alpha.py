# coding: utf-8

"""
Custom tasks to derive JER SF.
"""

import law
import order as od

from columnflow.util import maybe_import
from columnflow.tasks.framework.base import Requirements

from dijet.tasks.base import HistogramsBaseTask
from dijet.tasks.asymmetry import Asymmetry
from dijet.tasks.correlated_fit import CorrelatedFit

hist = maybe_import("hist")
np = maybe_import("numpy")
it = maybe_import("itertools")


class AlphaExtrapolation(
    HistogramsBaseTask,
    CorrelatedFit,
    law.LocalWorkflow,
):
    """
    Task to perform alpha extrapolation.
    Read in and plot asymmetry histograms.
    Extrapolate sigma_A( alpha->0 ).

    Processing steps:
    - read in prepared asymmetry distributions from `Asymmetry` task
    - extract asymmetry widths
    - perform extrapolation of widths to alpha=0 via linear fit including
      correlations
    """

    # declare output collection type and keys
    output_collection_cls = law.NestedSiblingFileCollection
    output_base_keys = ("widths", "extrapolation")

    # how to create the branch map
    branching_type = "separate"

    # upstream requirements
    reqs = Requirements(
        Asymmetry=Asymmetry,
    )

    #
    # methods required by law
    #

    def output(self):
        """Output has same structure as `Asymmetry` task."""
        return self.reqs.Asymmetry.output(self)

    def requires(self):
        """Require `Asymmetry` task."""
        return self.reqs.Asymmetry.req(self)

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["key"] = self.requires_from_branch()
        return reqs

    #
    # helper methods for handling task inputs/outputs
    #

    def load_input(self, key: str, level: str):
        return self.input()[key][self.branch_data.sample][level].load(formatter="pickle")

    def dump_output(self, key: str, level: str, obj: object):
        if key not in self.output_base_keys:
            raise ValueError(
                f"output key '{key}' not registered in "
                f"`{self.task_family}.output_base_keys`",
            )
        self.output()[key][self.branch_data.sample][level].dump(obj, formatter="pickle")

    #
    # task implementation
    #

    def _run_impl(self, datasets: list[od.Dataset], level: str, variable: str):
        """
        Implementation of width extrapolation from asymmetry distributions.
        """
        # check provided level
        if level not in ("gen", "reco"):
            raise ValueError(f"invalid level '{level}', expected one of: gen,reco")

        # dict storing either variables or their gen-level equivalents
        # for convenient access
        vars_ = self._make_var_lookup(level=level)

        #
        # start main processing
        #

        h_asyms = self.load_input("asym_cut", level=level)
        h_nevts = self.load_input("nevt", level=level)

        # Get widths of asymmetries
        asyms = h_asyms.axes[vars_["asymmetry"]].centers

        # Take mean value from normalized asymmetry
        axes_names = [a.name for a in h_asyms.axes]
        assert axes_names[-1] == vars_["asymmetry"], "asymmetry axis must come last"
        means = np.nansum(
            asyms * h_asyms.view().value,
            axis=-1,
            keepdims=True,
        )
        h_stds = h_asyms.copy()
        h_stds = h_stds[{vars_["asymmetry"]: sum}]

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
        self.dump_output("widths", level=level, obj=results_widths)

        # Get max alpha for fit; usually 0.3
        amax = 0.3  # TODO: define in config
        h_stds = h_stds[{vars_["alpha"]: slice(0, hist.loc(amax))}]
        h_nevts = h_nevts[{vars_["alpha"]: slice(0, hist.loc(amax))}]
        # exclude 0, the first bin, from alpha edges
        alphas = h_stds.axes[vars_["alpha"]].edges[1:]

        # TODO: More efficient procedure than for loop?
        #       - Idea: Array with same shape but with tuple (width, error) as entry
        n_bins = [
            len(h_stds.axes[bv].centers)
            for bv in vars_["binning"].values()
        ]
        n_methods = len(h_stds.axes["category"].centers)  # ony length
        inter = h_stds.copy().values()
        inter = inter[:, :2, ...]  # keep first two entries
        slope = h_stds.copy().values()
        slope = slope[:, :2, ...]  # keep first two entries
        for m, *bv_indices in it.product(
            range(n_methods),
            *[range(n) for n in n_bins],
        ):
            h_slice = {
                "category": m,
            }
            h_slice.update({
                bv: bv_index
                for bv, bv_index in zip(
                    vars_["binning"].values(),
                    bv_indices,
                )
            })
            tmp = h_stds[h_slice]
            tmp_evts = h_nevts[h_slice]
            coeff, err = self.get_correlated_fit(wmax=alphas, std=tmp.values(), nevts=tmp_evts.values())
            inter[(m, slice(None), *bv_indices)] = [coeff[1], err[1]]
            slope[(m, slice(None), *bv_indices)] = [coeff[0], err[0]]

        # NOTE: store fits into hist.
        h_intercepts = h_stds.copy()
        # Remove axis for alpha for histogram
        h_intercepts = h_intercepts[{vars_["alpha"]: sum}]
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

        # store y-intercepts and slopes from linear fit
        # in identically-structured histograms
        results_extrapolation = {
            "intercepts": h_intercepts,
            "slopes": h_slopes,
        }
        self.dump_output("extrapolation", level=level, obj=results_extrapolation)

    def run(self):
        # process histograms for all applicable levels
        sample = self.branch_data.sample
        for level, variable in self.iter_levels_variables():
            print(f"performing alpha extrapolation for {sample = }, {level = }, {variable = }")
            self._run_impl(self.branch_data.datasets, level=level, variable=variable)
