# coding: utf-8

import law

# from dijet.tasks.base import HistogramsBaseTask
from columnflow.util import maybe_import, DotDict
from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.remote import RemoteWorkflow

from dijet.tasks.alpha import AlphaExtrapolation
from dijet.constants import eta
from dijet.plotting.base import PlottingBaseTask
from dijet.plotting.util import eta_bin, pt_bin, alpha_bin, add_text, dot_to_p

hist = maybe_import("hist")
np = maybe_import("numpy")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")


class PlotAsymmetries(
    PlottingBaseTask,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    """
    Task to plot all asymmetries.
    One plot for each eta, pt and alpha bin for each method (fe,sm).
    """

    output_collection_cls = law.NestedSiblingFileCollection

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
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
            branch=-1
        )

    def load_asymmetry(self):
        return (
            self.input().collection[0]["asym"].load(formatter="pickle"),
            self.input().collection[1]["asym"].load(formatter="pickle"),
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

    def plot_asymmetry(self, data, mc, asym):
        fig, ax = plt.subplots()
        plt.bar(
            asym.flatten(),
            mc.flatten(),
            align="center",
            width=np.diff(asym)[0],
            alpha=0.6,
            color="indianred",
            edgecolor="none",
            label="MC",
        )
        plt.scatter(asym.flatten(), data.flatten(), marker="o", color="black", label="Data")
        ax.set_xlabel("Asymmetry")
        ax.set_ylabel(r"$\Delta$N/N")
        return fig, ax

    def run(self):
        asymm_data, asymm_mc = self.load_asymmetry()

        eta_lo, eta_hi = self.branch_data.eta
        eta_midp = 0.5 * (eta_lo + eta_hi)
        asymm_data = asymm_data[{"probejet_abseta": hist.loc(eta_midp), "dijets_alpha": slice(0, hist.loc(0.3))}]
        asymm_mc = asymm_mc[{"probejet_abseta": hist.loc(eta_midp), "dijets_alpha": slice(0, hist.loc(0.3))}]

        asymm_data.view().value = np.nan_to_num(asymm_data.view().value, nan=0.0)
        asymm_data.view().variance = np.nan_to_num(asymm_data.view().variance, nan=0.0)
        asymm_mc.view().value = np.nan_to_num(asymm_mc.view().value, nan=0.0)
        asymm_mc.view().variance = np.nan_to_num(asymm_mc.view().variance, nan=0.0)

        pt_edges = asymm_data.axes["dijets_pt_avg"].edges
        alpha_edges = asymm_data.axes["dijets_alpha"].edges
        asym_centers = asymm_data.axes["dijets_asymmetry"].centers

        # Set plotting style
        plt.style.use(mplhep.style.CMS)

        pos_x = 0.05
        pos_y = 0.95
        offset = 0.05

        text_eta_bin = eta_bin(eta_lo, eta_hi)
        store_bin_eta = f"eta_{dot_to_p(eta_lo)}_{dot_to_p(eta_hi)}"

        for m in self.LOOKUP_CATEGORY_ID:
            for ip, (pt_lo, pt_hi) in enumerate(zip(pt_edges[:-1], pt_edges[1:])):
                for ia, a in enumerate(alpha_edges[1:]):  # skip first alpha bin for nameing scheme
                    # TODO: status/debugging option for input to print current bin ?
                    # print(f"Start with pt {pt_lo} to {pt_hi} and alpha {a}")

                    # TODO: Include errors
                    input_ = {
                        "data": asymm_data[hist.loc(self.LOOKUP_CATEGORY_ID[m]), ia, ip, :].values(),
                        "mc": asymm_mc[hist.loc(self.LOOKUP_CATEGORY_ID[m]), ia, ip, :].values(),
                        "asym": asym_centers,
                    }

                    fig, ax = self.plot_asymmetry(**input_)
                    mplhep.cms.label(
                        lumi=41.48,  # TODO: from self.config_inst.x.luminosity?
                        com=13,
                        ax=ax,
                        llabel="Private Work",
                        data=True,
                    )

                    add_text(ax, pos_x, pos_y, text_eta_bin)
                    add_text(ax, pos_x, pos_y, pt_bin(pt_lo, pt_hi), offset=offset)
                    add_text(ax, pos_x, pos_y, alpha_bin(a), offset=2 * offset)

                    plt.xlim(-0.5, 0.5)
                    plt.legend(loc="upper right")

                    # keep short lines
                    store_bin_pt = f"pt_{dot_to_p(pt_lo)}_{dot_to_p(pt_hi)}"
                    store_bin_alpha = f"alpha_lt_{dot_to_p(a)}"
                    self.output()["single"].child(
                        f"asym_{m}_{store_bin_eta}_{store_bin_pt}_{store_bin_alpha}.pdf",
                        type="f",
                    ).dump(plt, formatter="mpl")
                    plt.close(fig)
