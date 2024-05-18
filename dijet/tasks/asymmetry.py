# coding: utf-8

"""
Custom tasks to derive JER SF.
"""

import law

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.histograms import MergeHistograms
from columnflow.util import maybe_import

from dijet.tasks.base import HistogramsBaseTask
from dijet.tasks.correlated_fit import CorrelatedFit

ak = maybe_import("awkward")
hist = maybe_import("hist")
np = maybe_import("numpy")


class Asymmetry(
    HistogramsBaseTask,
):
    """
    Task to prepare asymmetries for width extrapolation.
    Read in and plot asymmetry histograms.
    Cut of non-gaussian tails.
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
            "asym": target.child("asym.pickle", type="f"),
            "nevt": target.child("nevt.pickle", type="f"),
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
        # Skip over-/underflow bins (i.e. TH1F -> ComputeIntegral for UHH2)
        # h_all[{"dijets_asymmetry": sum}] includes such bins
        integral = h_all.values().sum(axis=axes_names.index("dijets_asymmetry"), keepdims=True)
        
        # Store for width extrapolation
        h_nevts = h_all.copy()
        h_nevts = h_nevts[{"dijets_asymmetry": sum}]
        h_nevts.view().value = np.squeeze(integral)
        self.output()["nevt"].dump(h_nevts, formatter="pickle")

        # normalize histogram to integral over asymmetry
        view.value = view.value / integral
        view.variance = view.variance / integral**2

        # Store in pickle file for plotting task
        self.output()["asym"].dump(h_all, formatter="pickle")
