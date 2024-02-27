# coding: utf-8

"""
Custom tasks to derive JER SF.
"""

import law

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.histograms import MergeHistograms
from columnflow.util import maybe_import

from dijet.tasks.base import HistogramsBaseTask

ak = maybe_import("awkward")
hist = maybe_import("hist")
np = maybe_import("numpy")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
it = maybe_import("itertools")


class AlphaExtrapolation(HistogramsBaseTask):
    """
    Task to perform alpha extrapolation.
    Read in and plot asymmetry histograms.
    Cut of non-gaussian tails and extrapolate sigma_A( alpha->0 ).
    """

    # Add nested sibling directories to output path
    output_collection_cls = law.NestedSiblingFileCollection

    # upstream requirements
    reqs = Requirements(
        MergeHistograms=MergeHistograms,
    )

    def requires(self):
        return {
            d: self.reqs.MergeHistograms.req(
                self,
                dataset=d,
                branch=-1,
                _exclude={"branches"},
                _prefer_cli={"variables"},
            )
            for d in self.datasets
        }

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["merged_hists"] = self.requires_from_branch()
        return reqs

    def load_histogram(self, dataset, variable):
        histogram = self.input()[dataset]["collection"][0]["hists"].targets[variable].load(formatter="pickle")
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
            "asym": target.child("asym.pickle", type="f", optional=True),
        }
        return outp

    def run(self):
        # TODO: Gen level for MC
        #       Correlated fit (in jupyter)

        h_all = []
        datasets, isMC = self.get_datasets()

        for dataset in datasets:
            for variable in self.variables:
                h_in = self.reduce_histogram(
                    self.load_histogram(dataset, variable),
                    self.processes,
                    self.shift,
                )
                # Store all hists in a list to sum over after reading
                h_all.append(h_in)
        h_all = sum(h_all)

        # TODO: Need own task to store asymmetry before this one
        #       New structure of base histogram task necessary
        axes_names = [a.name for a in h_all.axes]
        view = h_all.view()

        # replace histogram contents with cumulative sum over alpha bins
        view.value = np.apply_along_axis(np.cumsum, axis=axes_names.index("dijets_alpha"), arr=view.value)
        view.variance = np.apply_along_axis(np.cumsum, axis=axes_names.index("dijets_alpha"), arr=view.variance)

        # Get integral of asymmetries as array
        integral = h_all.values().sum(axis=axes_names.index("dijets_asymmetry"), keepdims=True)

        # normalize histogram to integral over asymmetry
        view.value = view.value / integral
        view.variance = view.variance / integral**2

        # Store in pickle file for plotting task
        self.output()["asym"].dump(h_all, formatter="pickle")

        # Get widths of asymmetries
        asyms = h_all.axes["dijets_asymmetry"].centers
        # Take mean value from normalized asymmetry
        assert axes_names[-1] == "dijets_asymmetry", "asymmetry axis must come last"
        means = np.nansum(
            asyms * h_all.view().value,
            axis=-1,
            keepdims=True,
        )
        h_stds = h_all.copy()
        h_stds = h_stds[{"dijets_asymmetry": sum}]
        # Get stds
        h_stds.view().value = np.sqrt(
            np.average(
                ((asyms - means)**2),
                weights=h_all.view().value,
                axis=-1,
            ),
        )
        # Get stds error; squeeze to reshape integral from (x,y,z,1) to (x,y,z)
        # note: error on std deviation analogous to implementation in ROOT::TH1
        # https://root.cern/doc/v630/TH1_8cxx_source.html#l07520
        h_stds.view().variance = h_stds.view().value**2 / (2 * np.squeeze(integral))

        # Store alphas here to get alpha up to 1
        # In the next stepts only alpha<0.3 needed; avoid slicing from there
        results_widths = {
            "widths": h_stds,
        }
        self.output()["widths"].dump(results_widths, formatter="pickle")

        # Get max alpha for fit; usually 0.3
        amax = 0.3  # TODO: define in config
        h_stds = h_stds[{"dijets_alpha": slice(0, hist.loc(amax))}]

        # TODO: Use correlated fit
        def fit_linear(subarray):
            alphas = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
            fitting = ~np.isnan(subarray)
            if len(subarray[fitting]) < 2:
                coefficients = [0, 0]
            else:
                coefficients = np.polyfit(alphas[fitting], subarray[fitting], 1)
            return coefficients
        # TODO: save fit results (chi2, ndf, etc.) for diagnostic
        fits = np.apply_along_axis(fit_linear, axis=axes_names.index("dijets_alpha"), arr=h_stds.view().value)

        # NOTE: store fits into hist.
        h_intercepts = h_stds.copy()
        # Remove axis for alpha for histogram
        h_intercepts = h_intercepts[{"dijets_alpha": sum}]
        # y intercept of fit (x=0)
        h_intercepts.view().value = fits[:, 0, :, :]
        # Errors temporarly used; Later get:
        # Error on fit from fit function (how?) or new method with three fits
        h_intercepts.view().variance = np.ones(h_intercepts.shape)

        h_slopes = h_fits.copy()
        # Slope of fit stored in index 1
        h_slopes.view().value = fits[:, 1, :, :]
        # Only stored for plotting, no defined error
        h_slopes.view().variance = np.zeros(h_fits.shape)

        results_extrapolation = {
            "intercepts": h_intercepts,
            "slopes": h_slopes,
        }
        self.output()["extrapolation"].dump(results_extrapolation, formatter="pickle")
