# coding: utf-8

"""
Custom tasks to derive JER SF.
"""

import law

from dijet.tasks.base import HistogramsBaseTask
from columnflow.util import maybe_import
from columnflow.tasks.framework.base import Requirements

from dijet.tasks.alpha import AlphaExtrapolation

ak = maybe_import("awkward")
hist = maybe_import("hist")
np = maybe_import("numpy")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
it = maybe_import("itertools")


class JER(HistogramsBaseTask):
    """
    Task to perform alpha extrapolation.
    Read in and plot asymmetry histograms.
    Cut of non-gaussian tails and extrapolate sigma_A( alpha->0 ).
    """

    output_collection_cls = law.NestedSiblingFileCollection

    # upstream requirements
    reqs = Requirements(
        AlphaExtrapolation=AlphaExtrapolation,
    )

    def requires(self):
        return self.reqs.AlphaExtrapolation.req(
            self,
            branch=-1,
            _exclude={"branches"},
        )

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["key"] = self.as_branch().requires()
        return reqs

    def load_extrapolation(self):
        histogram = self.input().collection[0]["extrapolation"].load(formatter="pickle")
        return histogram

    def output(self) -> dict[law.FileSystemTarget]:
        # TODO: Into base and add argument alphas, jers, etc.
        sample = self.extract_sample()
        target = self.target(f"{sample}", dir=True)
        outp = {
            "jers": target.child("jers.pickle", type="f"),
        }
        return outp

    def run(self):
        results_extrapolation = self.load_extrapolation()

        # ### Now JER SM Data
        widths = results_extrapolation["intercepts"].copy()

        # Get index of method to change methods individually
        # hist.view() only works if the full histgram is taken w.o. selecting the categories before hand
        category_id = {"sm": 1, "fe": 2}
        # Get list of categories with the correct order
        categories = list(jer.axes["category"])
        index_methods = {m: categories.index(category_id[m]) for m in category_id}

        # calcuate jer for standard method
        view.value[index_methods["sm"]] = view.value[index_methods["sm"], :, :] * np.sqrt(2)

        # TODO: Define eta bin in config
        # NOTE: weighting number by appearence in eta regions
        jer_ref = np.mean(view.value[index_methods["sm"], :5, 0], axis=0).reshape(1, -1)
        view.value[index_methods["fe"]] = np.sqrt(4 * view.value[index_methods["fe"], :, :]**2 - jer_ref**2)

        results_jers = {
            "jer": jer,
        }
        self.output()["jers"].dump(results_jers, formatter="pickle")
