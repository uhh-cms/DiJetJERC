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
        init=0