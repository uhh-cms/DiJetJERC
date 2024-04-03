# coding: utf-8

import law

# from dijet.tasks.base import HistogramsBaseTask
from columnflow.util import maybe_import, DotDict
from columnflow.tasks.framework.base import Requirements

from dijet.tasks.alpha import AlphaExtrapolation
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

    def run(self):
        init=0