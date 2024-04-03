# coding: utf-8

import law

# from dijet.tasks.base import HistogramsBaseTask
from columnflow.util import maybe_import, DotDict
from columnflow.tasks.framework.base import Requirements

from dijet.tasks.jer import JER
from dijet.constants import eta
from dijet.plotting.base import PlottingBaseTask
from dijet.plotting.util import eta_bin, add_text, dot_to_p

hist = maybe_import("hist")
np = maybe_import("numpy")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")

class PlotJERs(PlottingBaseTask):
    """
    Task to plot all JERs.
    One plot for each eta bin for each method (fe,sm).
    """

    output_collection_cls = law.NestedSiblingFileCollection

    reqs = Requirements(
        JER=JER,
    )

    def requires(self):
        return self.reqs.JER.req(
            self,
            processes=("qcd", "data"),
            branch=-1,
        )

    def load_jers(self):
        return (
            self.input().collection[0]["jers"].load(formatter="pickle"),
            self.input().collection[1]["jers"].load(formatter="pickle"),
        )

    def output(self) -> dict[law.FileSystemTarget]:
        # TODO: Unstable for changes like data_jetmet_X
        #       Make independent like in config datasetname groups
        sample = self.extract_sample()
        target = self.target(f"{sample}", dir=True)
        # declare the main target
        outp = {
            "dummy": target.child("dummy.txt", type="f"),
        }
        return outp

    def run(self):
        jers_da, jers_mc = self.load_jers()
        jers_da = jers_da["jer"]
        jers_mc = jers_mc["jer"]

        jers_da.view().value = np.nan_to_num(jers_da.view().value, nan=0.0)
        jers_mc.view().value = np.nan_to_num(jers_mc.view().value, nan=0.0)
        jers_da.view().variance = np.nan_to_num(jers_da.view().variance, nan=0.0)
        jers_mc.view().variance = np.nan_to_num(jers_mc.view().variance, nan=0.0)

