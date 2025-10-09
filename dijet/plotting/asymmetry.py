# coding: utf-8
"""
Task for plotting asymmetry distributions.
"""
from __future__ import annotations

import itertools
import law

from functools import partial

from columnflow.util import maybe_import

from dijet.tasks.asymmetry import Asymmetry
from dijet.plotting.base import PlottingBaseTask
from dijet.plotting.util import annotate_corner, get_bin_slug, get_bin_label, plot_xy

hist = maybe_import("hist")
np = maybe_import("numpy")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")


class PlotAsymmetries(
    PlottingBaseTask,
):
    """
    Task to plot asymmetry distributions.

    Shows the distribution of the `--asymmetry-variable` for all given `--samples`
    and `--levels`. One plot is produced for each eta, pt and alpha bin
    for each method (fe, sm).
    The methods to take into account are given as `--categories`.
    """

    # how to create the branch map
    branching_type = "merged"

    # upstream workflow
    input_task_cls = Asymmetry

    # keys for looking up input task results
    input_keys = ("asym", "quantile")

    # plot file names will start with this
    output_prefix = "asym"

    # plot configuration (e.g. axes limits/labels/scales)
    plot_settings = {
        "legend_kwargs": dict(loc="upper right"),
        "xlabel": lambda self, ctx: self.config_inst.get_variable(ctx["vars"]["reco"]["asymmetry"]).x_title,
        "xlim": (-0.5, 0.5),
        "xscale": "linear",
        "ylabel": r"$\Delta$N/N",
        "ylim": (5e-5, 10),
        "yscale": "log",
    }

    #
    # helper methods for handling task inputs/outputs
    #

    def load_input(self, key: str, sample: str, level: str):
        coll_keys = [
            coll_key
            for coll_key, coll in self.input()["collection"].targets.items()
            if sample in coll[key]
        ]
        if len(coll_keys) != 1:
            raise RuntimeError(
                f"found {len(coll_keys)} input collections corresponding to "
                f"sample '{sample}', expected 1",
            )
        return self.input()["collection"][coll_keys[0]][key][sample][level].load(formatter="pickle")

    def load_inputs(self):
        return {
            key: {
                sample: {
                    level: self.load_input(key, sample=sample, level=level)
                    for level in self.levels
                    if level == "reco" or sample != "data"
                }
                for sample in self.samples
            }
            for key in self.input_keys
        }

    #
    # task implementation
    #

    def run(self):
        # load all inputs
        raw_inputs = self.load_inputs()

        # dict storing either variables or their gen-level equivalents
        # for convenient access
        vars_ = {
            level: self._make_var_lookup(level=level)
            for level in self.levels
        }

        # prepare inputs (clean nans)
        def _prepare_input(histogram):
            # map `nan` values to zero
            v = histogram.view()
            v.value = np.nan_to_num(v.value, nan=0.0)
            v.variance = np.nan_to_num(v.variance, nan=0.0)
            # return prepared histogram
            return histogram
        inputs = law.util.map_struct(_prepare_input, raw_inputs)

        # binning information from first histogram object
        # (assume identical binning for all)
        def _iter_flat(d: dict):
            if not isinstance(d, dict):
                yield d
                return
            for k, v in d.items():
                yield from _iter_flat(v)
        ref_object = next(_iter_flat(inputs["asym"]))

        # binning information from inputs
        alpha_edges = ref_object.axes[vars_["reco"]["alpha"]].edges
        binning_variable_edges = {
            bv: ref_object.axes[bv_resolved].edges
            for bv, bv_resolved in vars_["reco"]["binning"].items()
            if bv != "probejet_abseta"
        }

        # binning information from branch
        eta_lo, eta_hi = self.branch_data.eta
        eta_midp = 0.5 * (eta_lo + eta_hi)

        def iter_bins(edges, **add_kwargs):
            for i, (e_dn, e_up) in enumerate(zip(edges[:-1], edges[1:])):
                yield {"up": e_up, "dn": e_dn, **add_kwargs}

        # loop through bins and do plotting
        plt.style.use(mplhep.style.CMS)
        for m, (ia, alpha_up), *bv_bins in itertools.product(
            self.method_categories,
            enumerate(alpha_edges[1:]),
            *[iter_bins(bv_edges, var_name=bv) for bv, bv_edges in binning_variable_edges.items()],
        ):
            # initialize figure and axes
            fig, ax = plt.subplots()
            mplhep.cms.label(
                lumi=round(0.001 * self.config_inst.x.luminosity.get("nominal"), 2),  # /pb -> /fb
                com=f"{self.config_inst.campaign.ecm:g}",
                ax=ax,
                llabel="Private Work",
                data=True,
            )

            # selector to get current bin
            bin_selectors = {}
            for level in self.levels:
                bin_selectors[level] = {
                    "category": hist.loc(self.config_inst.get_category(m).name),
                    vars_[level]["alpha"]: hist.loc(alpha_up - 0.001),
                    vars_[level]["binning"]["probejet_abseta"]: hist.loc(eta_midp),
                }
                for bv_bin in bv_bins:
                    bin_selectors[level][vars_[level]["binning"][bv_bin["var_name"]]] = (
                        hist.loc(0.5 * (bv_bin["up"] + bv_bin["dn"]))
                    )

            # loop through samples/levels
            for sample, level in itertools.product(self.samples, self.levels):
                # only use reco level for data
                if sample == "data" and level != "reco":
                    continue

                # get input histogram for asymmetries
                h_in = inputs["asym"][sample][level]
                h_sliced = h_in[bin_selectors[level]]

                # look up plotting kwargs under samples
                plot_kwargs = dict(
                    self.config_inst.x("samples", {})
                    .get(sample, {}).get("plot_kwargs", {}),
                )
                # resolve task-specific kwargs
                plot_kwargs = plot_kwargs.get(
                    self.__class__,
                    plot_kwargs.get("__default__", {}),
                )
                # adjust for gen-level
                if level == "gen":
                    plot_kwargs = dict(plot_kwargs, **{
                        "color": "forestgreen",
                        "label": "MC (gen)",
                        "method": "step",
                        "where": "mid",
                    })

                # plot asymmetry distribution
                plot_xy(
                    h_sliced.axes[vars_[level]["asymmetry"]].centers,
                    h_sliced.values(),
                    xerr=h_sliced.axes[vars_[level]["asymmetry"]].widths / 2,
                    yerr=np.sqrt(h_sliced.variances()),
                    ax=ax,
                    **plot_kwargs,
                )

                # plot quantiles as vertical lines
                hs_in_quantiles = inputs["quantile"][sample][level]
                q_lo = hs_in_quantiles["low"][bin_selectors[level]].value
                q_up = hs_in_quantiles["up"][bin_selectors[level]].value
                ax.axvline(q_lo, 0, 0.67, color=plot_kwargs.get("color"), linestyle="--")
                ax.axvline(q_up, 0, 0.67, color=plot_kwargs.get("color"), linestyle="--")

            #
            # annotations
            #

            # texts to display on plot
            annotation_texts = [
                # category label
                self.config_inst.get_category(m).label,
                # alpha bin
                get_bin_label(self.alpha_variable_inst, (0, alpha_up)),
                # eta bin
                get_bin_label(self.binning_variable_insts["probejet_abseta"], (eta_lo, eta_hi)),
            ]

            # texts for other binning variables
            for i, bv_bin in enumerate(bv_bins):
                bin_edges = (bv_bin["dn"], bv_bin["up"])
                bin_label = get_bin_label(self.binning_variable_insts[bv_bin["var_name"]], bin_edges)
                annotation_texts.append(bin_label)

            # curry function for convenience
            annotate = partial(annotate_corner, ax=ax, loc="upper left")

            # add annotations to plot
            for i, text in enumerate(annotation_texts):
                annotate(
                    text=text,
                    xy_offset=(20, -20 - 30 * i),
                )

            # figure adjustments
            self.apply_plot_settings(ax=ax, context={"vars": vars_})

            # legend
            ax.legend(**self.plot_settings.get("legend_kwargs", {}))

            # compute plot filename
            fname_parts = [
                # base name
                self.output_prefix,
                # method
                m,
                # alpha
                get_bin_slug(self.alpha_variable_inst, (0, alpha_up)),
                # abseta bin
                get_bin_slug(self.binning_variable_insts["probejet_abseta"], (eta_lo, eta_hi)),
            ]
            # other bins
            for bv_bin in bv_bins:
                bin_edges = (bv_bin["dn"], bv_bin["up"])
                bin_slug = get_bin_slug(self.binning_variable_insts[bv_bin["var_name"]], bin_edges)
                fname_parts.append(bin_slug)

            # save plot to file
            self.save_plot("__".join(fname_parts), fig)
            plt.close(fig)
