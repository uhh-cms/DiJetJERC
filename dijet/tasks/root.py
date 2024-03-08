# coding: utf-8

import law

from dijet.tasks.base import HistogramsBaseTask
from columnflow.util import maybe_import, import_ROOT
from columnflow.tasks.framework.base import Requirements

from dijet.tasks.jer import JER

ak = maybe_import("awkward")
hist = maybe_import("hist")
np = maybe_import("numpy")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
it = maybe_import("itertools")
up = maybe_import("uproot")


class JERtoRoot(HistogramsBaseTask):
    """
    Task to convert JER output to rootfiles.
    It is not yet possible to write TGraphs via uproot.
    Efforts by the uproot Team ongoing. Related PR:
    https://github.com/scikit-hep/uproot5/pull/1144
    """

    output_collection_cls = law.NestedSiblingFileCollection

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
        reqs["key"] = self.as_branch().requires()
        return reqs

    def load_jer(self):
        histogram = self.input().collection[0]["jers"].load(formatter="pickle")
        return histogram

    def output(self) -> dict[law.FileSystemTarget]:
        sample = self.extract_sample()
        target = self.target(f"{sample}", dir=True)
        outp = {
            "jers": target.child("jers.pickle", type="f"),
        }
        return outp

    def run(self):
        results_jer = self.load_jer()
        jer = results_jer["jer"]

        eta_bins = jer.axes["probejet_abseta"].edges
        eta_centers = jer.axes["probejet_abseta"].centers

        # Get error for pt bins. These will be symmetrical for the moment
        # TODO: Define pt value by the mean value of the pt in given a pt bin
        pt_bins = jer.axes["dijets_pt_avg"].edges
        pt_centers = jer.axes["dijets_pt_avg"].centers
        pt_error_low = pt_centers - pt_bins[:-1]
        pt_error_high = pt_bins[1:] - pt_centers

        def get_eta_bins(self, i: int):
            up = "{:.3f}".format(eta_bins[i])
            do = "{:.3f}".format(eta_bins[i+1])
            bin_low = up.replace(".","p")
            bin_high = do.replace(".","p")
            return bin_low, bin_high

        # Store values for TGraphAsymmError in dictionary and convert with additional script.
        results_jers = {}
        for m in self.category_id:
            for i in range(len(eta_bins)-1):
                low, high = get_eta_bins(self, i)
                tmp = jer[
                    {
                        "category": hist.loc(self.category_id[m]),
                        "probejet_abseta": i,
                    }
                ]
                jers = tmp.values()
                jers_errors = tmp.variances()
                assert len(jers)==len(pt_centers), f"Check number of bins for {i}"
                results_jers[f"jer_eta_{low}_{high}_{m}"] = {
                    "fY": jers,
                    "fYerrUp": jers_errors,
                    "fYerrDown": jers_errors,
                    "fX": pt_centers,
                    "fXerrUp": pt_error_high,
                    "fXerrDown": pt_error_low,
                }

        self.output()["jers"].dump(results_jers, formatter="pickle")
