# coding: utf-8

import law

from dijet.tasks.base import HistogramsBaseTask
from columnflow.util import maybe_import
from columnflow.tasks.framework.base import Requirements

from dijet.tasks.jer import JER

hist = maybe_import("hist")


class JERtoRoot(HistogramsBaseTask):
    """
    Task to convert JER output to rootfiles.
    It is not yet possible to write TGraphs via uproot.
    Efforts by the uproot Team ongoing. Related PR:
    https://github.com/scikit-hep/uproot5/pull/1144
    """

    output_collection_cls = law.NestedSiblingFileCollection

    # how to create the branch map
    branching_type = JER.branching_type

    # upstream requirements
    reqs = Requirements(
        JER=JER,
    )

    # TODO: write base task for root conversion
    #       keep HistogramBaseTask as long as no conversion possible
    def requires(self):
        return self.reqs.JER.req(
            self,
            branch=-1,
            _exclude={"branches"},
        )

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["key"] = self.requires_from_branch()
        return reqs

    def load_jer(self):
        histogram = self.input().collection[0]["jers"].load(formatter="pickle")
        return histogram

    def output(self) -> dict[law.FileSystemTarget]:
        outp = {
            "jers": self.target("jers.pickle", type="f"),
        }
        return outp

    def run(self):
        results_jer = self.load_jer()
        h_jer = results_jer["jer"]

        # get pt bins (errors symmetrical for now)
        pt_bins = h_jer.axes[self._vars.pt].edges
        pt_centers = h_jer.axes[self._vars.pt].centers
        pt_error_lo = pt_centers - pt_bins[:-1]
        pt_error_hi = pt_bins[1:] - pt_centers

        # compute and store values for building `TGraphAsymmError`
        # in external script
        results_jers = {}
        abseta_bins = h_jer.axes[self._vars.abseta].edges
        for method in self.LOOKUP_CATEGORY_ID:
            for abseta_lo, abseta_hi in zip(
                abseta_bins[:-1],
                abseta_bins[1:],
            ):

                abseta_lo_str = f"{abseta_lo:.3f}".replace(".", "p")
                abseta_hi_str = f"{abseta_hi:.3f}".replace(".", "p")
                abseta_str = f"abseta_{abseta_lo_str}_{abseta_hi_str}"

                abseta_center = (abseta_lo + abseta_hi) / 2
                h_jer_category_abseta = h_jer[
                    {
                        "category": hist.loc(self.LOOKUP_CATEGORY_ID[method]),
                        self._vars.abseta: hist.loc(abseta_center),
                    }
                ]

                jers = h_jer_category_abseta.values()
                jers_errors = h_jer_category_abseta.variances()

                # sanity check
                assert len(jers) == len(pt_centers), f"check number of bins for {abseta_str!r}"

                # store results
                results_jers[f"jer_{abseta_str}_{method}"] = {
                    "fY": jers,
                    "fYerrUp": jers_errors,
                    "fYerrDown": jers_errors,
                    "fX": pt_centers,
                    "fXerrUp": pt_error_hi,
                    "fXerrDown": pt_error_lo,
                }

        # store results
        self.output()["jers"].dump(results_jers, formatter="pickle")
