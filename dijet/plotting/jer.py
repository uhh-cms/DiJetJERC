# coding: utf-8
"""
Task for plotting jet energy resolution (JER).
"""
from __future__ import annotations

import itertools
import law

from functools import partial

from columnflow.util import maybe_import

from dijet.tasks.jer import JER
from dijet.plotting.base import PlottingBaseTask
from dijet.plotting.util import annotate_corner, get_bin_slug, get_bin_label

hist = maybe_import("hist")
np = maybe_import("numpy")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")


class PlotJERs(
    PlottingBaseTask,
):
    """
    Task to plot the JER extracted from the extrapolated widths.

    Shows the JER extracted from the width of the `--asymmetry-variable`
    for all given `--samples`. One plot is produced for each abseta bin
    for each method (fe, sm).
    The methods to take into account are given as `--categories`.
    """

    # how to create the branch map
    branching_type = "merged"

    # upstream workflow
    input_task_cls = JER

    #
    # helper methods for handling task inputs/outputs
    #

    def load_input(self, key: str, sample: str):
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
        return self.input()["collection"][coll_keys[0]][key][sample].load(formatter="pickle")

    #
    # task implementation
    #

    @staticmethod
    def _plot_shim(x, y, xerr=None, yerr=None, method=None, ax=None, **kwargs):
        """
        Draw one series of xy values.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = plt.gcf(), plt.gca()

        method = method or "errorbar"

        method_func = getattr(ax, method, None)
        if method_func is None:
            raise ValueError(f"invalid plot method '{method}'")

        if method == "bar":
            kwargs.update(
                align="center",
                width=2 * xerr,
                yerr=yerr,
            )
        elif method == "step":
            kwargs.pop("xerr", None)
            kwargs.pop("yerr", None)
            kwargs.pop("edgecolor", None)
        else:
            kwargs["xerr"] = xerr
            kwargs["yerr"] = yerr

        method_func(
            x.flatten(),
            y.flatten(),
            **kwargs,
        )

        return fig, ax

    def run(self):
        # load inputs (asymmetries and quantiles)
        raw_inputs = {
            key: {
                sample: self.load_input(key, sample=sample)
                for sample in self.samples
            }
            for key in ("jers",)
        }

        # dict storing either variables or their gen-level equivalents
        # for convenient access
        vars_ = self._make_var_lookup(level="reco")

        # prepare inputs (apply slicing, clean nans)
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
        ref_object = next(_iter_flat(inputs["jers"]))

        # binning information from inputs
        binning_variable_edges = {
            bv: ref_object.axes[bv_resolved].edges
            for bv, bv_resolved in vars_["binning"].items()
            if bv not in ("probejet_abseta", "dijets_pt_avg")
        }

        # binning information from branch
        eta_lo, eta_hi = self.branch_data.eta
        eta_midp = 0.5 * (eta_lo + eta_hi)

        def iter_bins(edges, **add_kwargs):
            for i, (e_dn, e_up) in enumerate(zip(edges[:-1], edges[1:])):
                yield {"up": e_up, "dn": e_dn, **add_kwargs}

        # loop through bins and do plotting
        plt.style.use(mplhep.style.CMS)
        for m, *bv_bins in itertools.product(
            self.LOOKUP_CATEGORY_ID,
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
            bin_selector = {}
            bin_selector = {
                "category": hist.loc(self.LOOKUP_CATEGORY_ID[m]),
                vars_["binning"]["probejet_abseta"]: hist.loc(eta_midp),
            }
            for bv_bin in bv_bins:
                bin_selector[vars_["binning"][bv_bin["var_name"]]] = (
                    hist.loc(0.5 * (bv_bin["up"] + bv_bin["dn"]))
                )

            # loop through samples
            for sample in self.samples:

                # get input histogram for widths
                h_in = inputs["jers"][sample]["jer"]
                h_sliced = h_in[bin_selector]

                # plot asymmetry distribution
                plot_kwargs = dict(
                    self.config_inst.x("samples", {})
                    .get(sample, {}).get("plot_kwargs", {}),
                )
                # resolve task-specific kwargs
                plot_kwargs = plot_kwargs.get(
                    self.__class__,
                    plot_kwargs.get("__default__", {}),
                )

                # plot JER
                self._plot_shim(
                    h_sliced.axes[vars_["binning"]["dijets_pt_avg"]].centers,
                    h_sliced.values(),
                    yerr=np.sqrt(h_sliced.variances()),
                    ax=ax,
                    **plot_kwargs,
                )

            #
            # annotations
            #

            # texts to display on plot
            annotation_texts = [
                # category label
                self.config_inst.get_category(m).label,
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
            # pt_edges = ref_object.axes["dijets_pt_avg"].edges
            # ax.set_xlim(pt_edges[0], pt_edges[-1])
            ax.set_xscale("log")
            ax.set_xlim(49, 2100)
            ax.set_ylim(0, 0.5)
            ax.legend(loc="upper right")
            ax.set_xlabel(
                self.config_inst.get_variable(vars_["binning"]["dijets_pt_avg"]).get_full_x_title(),
            )
            ax.set_ylabel(r"Jet energy resolution")

            # compute plot filename
            fname_parts = [
                # base name
                "jers",
                # method
                m,
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
