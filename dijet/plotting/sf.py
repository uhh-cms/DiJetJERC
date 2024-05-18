# coding: utf-8

import law

# from dijet.tasks.base import HistogramsBaseTask
from columnflow.util import maybe_import
from columnflow.tasks.framework.base import Requirements

from dijet.tasks.sf import SF
from dijet.plotting.base import PlottingBaseTask
from dijet.plotting.util import eta_bin, add_text, dot_to_p

hist = maybe_import("hist")
np = maybe_import("numpy")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")


class PlotSFs(PlottingBaseTask):
    """
    Task to plot all SFs.
    One plot for each eta bin for each method (fe,sm).
    """

    output_collection_cls = law.NestedSiblingFileCollection

    reqs = Requirements(
        SF=SF,
    )

    def requires(self):
        return self.reqs.SF.req(
            self,
            processes=("qcd", "data"),
            _exclude={"branches"},
        )

    def load_sfs(self):
        return self.input()["sfs"].load(formatter="pickle")["sfs"]

    def output(self) -> dict[law.FileSystemTarget]:
        # TODO: Unstable for changes like data_jetmet_X
        #       Make independent like in config datasetname groups
        sample = self.extract_sample()
        target = self.target(f"{sample}", dir=True)
        # declare the main target
        outp = {
            "single": target.child("sfs", type="d"),
            "dummy": target.child("dummy.txt", type="f"),
        }
        return outp

    def plot_sfs(self, sfs, pt):
        fig, ax = plt.subplots()

        plt.errorbar(pt, sfs["nom"], yerr=sfs["err"], fmt="o", color="black")

        ax.set_xlabel(r"$p_{T}^{ave}$")
        ax.set_ylabel(r"$SF$")
        return fig, ax

    def run(self):
        sfs = self.load_sfs()

        sfs.view().value = np.nan_to_num(sfs.view().value, nan=0.0)
        sfs.view().variance = np.nan_to_num(sfs.view().variance, nan=0.0)

        eta_edges = sfs.axes["probejet_abseta"].edges
        pt_centers = sfs.axes["dijets_pt_avg"].centers

        # Set plotting style
        plt.style.use(mplhep.style.CMS)

        pos_x = 0.05
        pos_y = 0.95
        for m in self.LOOKUP_CATEGORY_ID:
            for ie, (eta_lo, eta_hi) in enumerate(zip(eta_edges[:-1], eta_edges[1:])):

                input_ = {
                    "sfs": {
                        "nom": sfs[hist.loc(self.LOOKUP_CATEGORY_ID[m]), ie, :].values(),
                        "err": sfs[hist.loc(self.LOOKUP_CATEGORY_ID[m]), ie, :].variances(),
                    },
                    "pt": pt_centers,
                }

                fig, ax = self.plot_sfs(**input_)
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
                #plt.ylim(0.8, 20)
                ax.set_xscale("log")
                plt.legend(loc="upper right")

                store_bin_eta = f"eta_{dot_to_p(eta_lo)}_{dot_to_p(eta_hi)}"
                self.output()["single"].child(
                    f"sfs_{m}_{store_bin_eta}.pdf",
                    type="f",
                ).dump(plt, formatter="mpl")
                plt.close(fig)
