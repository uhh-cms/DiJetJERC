# coding: utf-8

import law

# from dijet.tasks.base import HistogramsBaseTask
from columnflow.util import maybe_import, DotDict
from columnflow.tasks.framework.base import Requirements

from dijet.tasks.alpha import AlphaExtrapolation
from dijet.constants import eta
from dijet.plotting.base import PlottingBaseTask
from dijet.plotting.util import eta_bin, pt_bin, add_text, dot_to_p

hist = maybe_import("hist")
np = maybe_import("numpy")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")


class PlotWidths(PlottingBaseTask):
    """
    Task to plot all alphas.
    One plot for each eta and pt bin for each method (fe,sm).
    """

    output_collection_cls = law.NestedSiblingFileCollection

    # upstream requirements
    reqs = Requirements(
        AlphaExtrapolation=AlphaExtrapolation,
    )

    def create_branch_map(self):
        """
        Workflow has one branch for each eta bin (eta).
        TODO: Hard coded for now
              Into Base Task
        """
        return [
            DotDict({"eta": (eta_lo, eta_hi)})
            for eta_lo, eta_hi in zip(eta[:-1], eta[1:])
        ]

    def requires(self):
        return self.reqs.AlphaExtrapolation.req(
            self,
            processes=("qcd", "data"),
            branch=-1,
        )

    def load_widths(self):
        return (
            self.input().collection[0]["widths"].load(formatter="pickle"),
            self.input().collection[1]["widths"].load(formatter="pickle"),
        )

    def load_extrapolation(self):
        return (
            self.input().collection[0]["extrapolation"].load(formatter="pickle"),
            self.input().collection[1]["extrapolation"].load(formatter="pickle"),
        )

    def output(self) -> dict[law.FileSystemTarget]:
        # TODO: Unstable for changes like data_jetmet_X
        #       Make independent like in config datasetname groups
        sample = self.extract_sample()
        target = self.target(f"{sample}", dir=True)
        # declare the main target
        eta_lo, eta_hi = self.branch_data.eta
        n_eta_lo = dot_to_p(eta_lo)
        n_eta_hi = dot_to_p(eta_hi)
        outp = {
            "single": target.child(f"eta_{n_eta_lo}_{n_eta_hi}", type="d"),
            "dummy": target.child(f"eta_{n_eta_lo}_{n_eta_hi}/dummy.txt", type="f"),
        }
        return outp

    def run(self):
        widths_da, widths_mc = self.load_widths()
        extrapol_da, extrapol_mc = self.load_extrapolation()

        eta_lo, eta_hi = self.branch_data.eta
        eta_midp = 0.5 * (eta_lo + eta_hi)

        # TODO: Variable as input for alpha to e.g. plot up to 0.5?
        upper = 0.3
        widths_da = widths_da["widths"][
            {"probejet_abseta": hist.loc(eta_midp), "dijets_alpha": slice(0, hist.loc(upper))}
        ]
        widths_mc = widths_mc["widths"][
            {"probejet_abseta": hist.loc(eta_midp), "dijets_alpha": slice(0, hist.loc(upper))}
        ]
        widths_da.view().value = np.nan_to_num(widths_da.view().value, nan=0.0)
        widths_mc.view().value = np.nan_to_num(widths_mc.view().value, nan=0.0)
        widths_da.view().variance = np.nan_to_num(widths_da.view().variance, nan=0.0)
        widths_mc.view().variance = np.nan_to_num(widths_mc.view().variance, nan=0.0)

        inter_da = extrapol_da["intercepts"][{"probejet_abseta": hist.loc(eta_midp)}]
        inter_mc = extrapol_mc["intercepts"][{"probejet_abseta": hist.loc(eta_midp)}]
        inter_da.view().value = np.nan_to_num(inter_da.view().value, nan=0.0)
        inter_mc.view().value = np.nan_to_num(inter_mc.view().value, nan=0.0)
        inter_da.view().variance = np.nan_to_num(inter_da.view().variance, nan=0.0)
        inter_mc.view().variance = np.nan_to_num(inter_mc.view().variance, nan=0.0)

        slope_da = extrapol_da["slopes"][{"probejet_abseta": hist.loc(eta_midp)}]
        slope_mc = extrapol_mc["slopes"][{"probejet_abseta": hist.loc(eta_midp)}]
        slope_da.view().value = np.nan_to_num(slope_da.view().value, nan=0.0)
        slope_mc.view().value = np.nan_to_num(slope_mc.view().value, nan=0.0)
        slope_da.view().variance = np.nan_to_num(slope_da.view().variance, nan=0.0)
        slope_mc.view().variance = np.nan_to_num(slope_mc.view().variance, nan=0.0)

        pt_edges = widths_da.axes["dijets_pt_avg"].edges
        alpha_edges = widths_da.axes["dijets_alpha"].edges[1:]

