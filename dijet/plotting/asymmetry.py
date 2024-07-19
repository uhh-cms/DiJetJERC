# coding: utf-8

import law

# from dijet.tasks.base import HistogramsBaseTask
from columnflow.util import maybe_import, DotDict
from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.remote import RemoteWorkflow

from dijet.tasks.asymmetry import Asymmetry
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
        Asymmetry=Asymmetry,
    )

    colors = {
        "da": "black",
        "mc": "indianred",
    }

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
        return self.reqs.Asymmetry.req(
            self,
            processes=("qcd", "data"),
            branch=-1,
        )

    def load_asymmetry(self):
        return (
            self.input().collection[0]["asym"].load(formatter="pickle"),
            self.input().collection[1]["asym"].load(formatter="pickle"),
        )

    def load_quantiles(self):
        return (
            self.input().collection[0]["quantiles"].load(formatter="pickle"),
            self.input().collection[1]["quantiles"].load(formatter="pickle"),
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

    def plot_asymmetry(self, content, error, asym):
        fig, ax = plt.subplots()
        plt.bar(
            asym.flatten(),
            content["mc"].flatten(),
            yerr=error["mc"].flatten(),
            align="center",
            width=np.diff(asym)[0],
            alpha=0.6,
            color=self.colors["mc"],
            edgecolor="none",
            label="MC",
        )
        plt.errorbar(
            asym.flatten(),
            content["da"].flatten(),
            yerr=error["da"].flatten(),
            fmt="o",
            marker="o",
            fillstyle="full",
            color=self.colors["da"],
            label="Data",
        )

        ax.set_xlabel("Asymmetry")
        ax.set_ylabel(r"$\Delta$N/N")
        ax.set_yscale("log")
        return fig, ax

    def run(self):
        asymm_da, asymm_mc = self.load_asymmetry()
        quant_da, quant_mc = self.load_quantiles()

        eta_lo, eta_hi = self.branch_data.eta
        eta_midp = 0.5 * (eta_lo + eta_hi)

        asymm_da = asymm_da[{"probejet_abseta": hist.loc(eta_midp), "dijets_alpha": slice(0, hist.loc(0.3))}]
        asymm_mc = asymm_mc[{"probejet_abseta": hist.loc(eta_midp), "dijets_alpha": slice(0, hist.loc(0.3))}]

        axis_eta = "probejet_abseta"  # to reduce line characters
        quant_da["low"] = quant_da["low"][{axis_eta: hist.loc(eta_midp), "dijets_alpha": slice(0, hist.loc(0.3))}]
        quant_mc["low"] = quant_mc["low"][{axis_eta: hist.loc(eta_midp), "dijets_alpha": slice(0, hist.loc(0.3))}]
        quant_da["up"] = quant_da["up"][{axis_eta: hist.loc(eta_midp), "dijets_alpha": slice(0, hist.loc(0.3))}]
        quant_mc["up"] = quant_mc["up"][{axis_eta: hist.loc(eta_midp), "dijets_alpha": slice(0, hist.loc(0.3))}]

        asymm_da.view().value = np.nan_to_num(asymm_da.view().value, nan=0.0)
        asymm_da.view().variance = np.nan_to_num(asymm_da.view().variance, nan=0.0)
        asymm_mc.view().value = np.nan_to_num(asymm_mc.view().value, nan=0.0)
        asymm_mc.view().variance = np.nan_to_num(asymm_mc.view().variance, nan=0.0)

        pt_edges = asymm_da.axes["dijets_pt_avg"].edges
        alpha_edges = asymm_da.axes["dijets_alpha"].edges
        asym_centers = asymm_da.axes["dijets_asymmetry"].centers

        # Set plotting style
        plt.style.use(mplhep.style.CMS)

        pos_x = 0.05
        pos_y = 0.95
        offset = 0.05
        x_lim = [-0.5, 0.5]
        y_lim = [0.00005, 10]
        range_quantile = [y_lim[0], y_lim[1] / 100]  # Adjust to y scale

        text_eta_bin = eta_bin(eta_lo, eta_hi)
        store_bin_eta = f"eta_{dot_to_p(eta_lo)}_{dot_to_p(eta_hi)}"

        for m in self.LOOKUP_CATEGORY_ID:
            for ip, (pt_lo, pt_hi) in enumerate(zip(pt_edges[:-1], pt_edges[1:])):
                for ia, a in enumerate(alpha_edges[1:]):  # skip first alpha bin for nameing scheme
                    # TODO: status/debugging option for input to print current bin ?
                    # print(f"Start with pt {pt_lo} to {pt_hi} and alpha {a}")

                    input_ = {
                        "content": {
                            "da": asymm_da[hist.loc(self.LOOKUP_CATEGORY_ID[m]), ia, ip, :].values(),
                            "mc": asymm_mc[hist.loc(self.LOOKUP_CATEGORY_ID[m]), ia, ip, :].values(),
                        },
                        "error": {
                            "da": np.sqrt(asymm_da[hist.loc(self.LOOKUP_CATEGORY_ID[m]), ia, ip, :].variances()),
                            "mc": np.sqrt(asymm_mc[hist.loc(self.LOOKUP_CATEGORY_ID[m]), ia, ip, :].variances()),
                        },
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

                    plt.xlim(x_lim[0], x_lim[1])
                    plt.ylim(y_lim[0], y_lim[1])
                    plt.legend(loc="upper right")

                    q_da_lo = quant_da["low"][hist.loc(self.LOOKUP_CATEGORY_ID[m]), ia, ip].value
                    q_da_up = quant_da["up"][hist.loc(self.LOOKUP_CATEGORY_ID[m]), ia, ip].value
                    q_mc_lo = quant_mc["low"][hist.loc(self.LOOKUP_CATEGORY_ID[m]), ia, ip].value
                    q_mc_up = quant_mc["up"][hist.loc(self.LOOKUP_CATEGORY_ID[m]), ia, ip].value
                    plt.plot([q_da_lo, q_da_lo], range_quantile, color=self.colors["da"], linestyle="--")
                    plt.plot([q_da_up, q_da_up], range_quantile, color=self.colors["da"], linestyle="--")
                    plt.plot([q_mc_lo, q_mc_lo], range_quantile, color=self.colors["mc"], linestyle="--")
                    plt.plot([q_mc_up, q_mc_up], range_quantile, color=self.colors["mc"], linestyle="--")

                    # keep short lines
                    store_bin_pt = f"pt_{dot_to_p(pt_lo)}_{dot_to_p(pt_hi)}"
                    store_bin_alpha = f"alpha_lt_{dot_to_p(a)}"
                    self.output()["single"].child(
                        f"asym_{m}_{store_bin_eta}_{store_bin_pt}_{store_bin_alpha}.pdf",
                        type="f",
                    ).dump(plt, formatter="mpl")
                    plt.close(fig)
