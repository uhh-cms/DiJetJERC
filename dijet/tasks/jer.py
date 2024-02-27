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
        # load extrapolation results
        results_extrapolation = self.load_extrapolation()

        # get extrapolated distribution widths
        h_widths = results_extrapolation["intercepts"]

        # get index on `category` axis corresponding to
        # the two computation methods
        category_id = {"sm": 1, "fe": 2}
        categories = list(h_widths.axes["category"])
        index_methods = {m: categories.index(category_id[m]) for m in category_id}

        # calcuate JER for standard method
        jer_sm_val = h_widths[index_methods["sm"], :, :].view().value * np.sqrt(2)
        jer_sm_err = np.sqrt(h_widths[index_methods["sm"], :, :].view().variance) * np.sqrt(2)

        # average over first few eta bins to get
        # reference JER for forward method
        # TODO: Define eta bin in config
        jer_ref_val = np.mean(jer_sm_val[:5, :], axis=0, keepdims=True)
        jer_ref_err = np.mean(jer_sm_err[:5, :], axis=0, keepdims=True)

        # calculate JER for forward extension method
        jer_fe_val = np.sqrt(4 * h_widths[index_methods["fe"], :, :].view().value**2 - jer_ref_val**2)
        term_probe = 2 * h_widths[index_methods["fe"], :, :].values() * h_widths[index_methods["fe"], :, :].variances()
        term_ref = jer_ref_val * jer_ref_err
        jer_fe_err = np.sqrt(term_probe**2 + term_ref**2) / jer_fe_val

        # create output histogram and view for filling
        h_jer = h_widths.copy()
        v_jer = h_jer.view()

        # write JER values to output histogram
        v_jer[index_methods["sm"], :, :].value = jer_sm_val
        v_jer[index_methods["sm"], :, :].variance = jer_sm_err**2
        v_jer[index_methods["fe"], :, :].value = jer_fe_val
        v_jer[index_methods["fe"], :, :].variance = jer_fe_err**2

        results_jers = {
            "jer": h_jer,
        }
        self.output()["jers"].dump(results_jers, formatter="pickle")
