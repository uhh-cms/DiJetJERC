# coding: utf-8

from __future__ import annotations

# import tabulate
import law

from columnflow.util import maybe_import

np = maybe_import("numpy")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
hist = maybe_import("hist")

logger = law.logger.get_logger(__name__)











# coding: utf-8

import law

from dijet.tasks.base import HistogramsBaseTask
from columnflow.util import maybe_import
from columnflow.tasks.framework.base import Requirements

from dijet.tasks.jer import JER

hist = maybe_import("hist")


class PlotAsymmetries(HistogramsBaseTask):
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

def plot_asymmetry(
        asymmetry,
        centers,
        eta, pt, alpha,
        output: law.FileSystemDirectoryTarget,
) -> None:
    # use CMS plotting style
    plt.style.use(mplhep.style.CMS)

    fig, ax = plt.subplots()
    plt.scatter(centers.flatten(), asymmetry.flatten(), marker="s", color="red", where="mid")
    plt.xlim(-0.6, 0.6)
    output["asym"].child(f"asym_e{eta}_p{pt}_a{alpha}.pdf", type="f").dump(plt, formatter="mpl")
