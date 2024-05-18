# coding: utf-8

import law

# from dijet.tasks.base import HistogramsBaseTask
from columnflow.util import maybe_import
from columnflow.tasks.framework.base import Requirements

from dijet.tasks.jer import JER
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

    colors = {
        "da": "black",
        "mc": "indianred",
    }

    def requires(self):
        return self.reqs.JER.req(
            self,
            processes=("qcd", "data"),
            _exclude={"branches"},
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
            "single": target.child("jers", type="d"),
            "dummy": target.child("dummy.txt", type="f"),
        }
        return outp

    def plot_jers(self, data, mc, pt):
        fig, ax = plt.subplots()

        plt.errorbar(pt, mc["nom"], yerr=mc["err"], fmt="o", color=self.colors["mc"], label="MC")
        plt.errorbar(pt, data["nom"], yerr=data["err"], fmt="o", color=self.colors["da"], label="Data")

        ax.set_xlabel(r"$p_{T}^{ave}$")
        ax.set_ylabel(r"$JER$")
        return fig, ax

    def run(self):
        jers_da, jers_mc = self.load_jers()
        jers_da = jers_da["jer"]
        jers_mc = jers_mc["jer"]

        jers_da.view().value = np.nan_to_num(jers_da.view().value, nan=0.0)
        jers_mc.view().value = np.nan_to_num(jers_mc.view().value, nan=0.0)
        jers_da.view().variance = np.nan_to_num(jers_da.view().variance, nan=0.0)
        jers_mc.view().variance = np.nan_to_num(jers_mc.view().variance, nan=0.0)

        eta_edges = jers_da.axes["probejet_abseta"].edges
        pt_centers = jers_da.axes["dijets_pt_avg"].centers

        # Set plotting style
        plt.style.use(mplhep.style.CMS)

        pos_x = 0.05
        pos_y = 0.95

        for m in self.LOOKUP_CATEGORY_ID:
            for ie, (eta_lo, eta_hi) in enumerate(zip(eta_edges[:-1], eta_edges[1:])):

                input_ = {
                    "data": {
                        "nom": jers_da[hist.loc(self.LOOKUP_CATEGORY_ID[m]), ie, :].values(),
                        "err": jers_da[hist.loc(self.LOOKUP_CATEGORY_ID[m]), ie, :].variances(),
                    },
                    "mc": {
                        "nom": jers_mc[hist.loc(self.LOOKUP_CATEGORY_ID[m]), ie, :].values(),
                        "err": jers_mc[hist.loc(self.LOOKUP_CATEGORY_ID[m]), ie, :].variances(),
                    },
                    "pt": pt_centers,
                }

                fig, ax = self.plot_jers(**input_)
                mplhep.cms.label(
                    lumi=41.48,  # TODO: from self.config_inst.x.luminosity?
                    com=13,
                    ax=ax,
                    llabel="Private Work",
                    data=True,
                )

                text_eta_bin = eta_bin(eta_lo, eta_hi)
                add_text(ax, pos_x, pos_y, text_eta_bin)
                print(f"Start with eta {text_eta_bin} for {m} method")

                plt.xlim(49, 2100)
                plt.ylim(0, 2)
                ax.set_xscale("log")
                plt.legend(loc="upper right")

                store_bin_eta = f"eta_{dot_to_p(eta_lo)}_{dot_to_p(eta_hi)}"
                self.output()["single"].child(
                    f"jers_{m}_{store_bin_eta}.pdf",
                    type="f",
                ).dump(plt, formatter="mpl")
                plt.close(fig)
