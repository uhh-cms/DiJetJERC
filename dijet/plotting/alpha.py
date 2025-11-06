# coding: utf-8
"""
Task for plotting asymmetry widths as a function of alpha
and their extrapolation to alpha = 0.
"""
from __future__ import annotations

import itertools
import law

from functools import partial

from columnflow.util import maybe_import

from dijet.tasks.alpha import AlphaExtrapolation
from dijet.plotting.base import PlottingBaseTask
from dijet.plotting.util import annotate_corner, plot_xy
from dijet.util import product_dict

logger = law.logger.get_logger(__name__)

hist = maybe_import("hist")
np = maybe_import("numpy")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")


class PlotAlphaExtrapolation(
    PlottingBaseTask,
):
    """
    Task to plot the extrapolation of asymmetry widths.

    Shows the width of the asymmetry distribution as a function of alpha,
    as well as the linear extrapolation, for all given `--samples`
    and `--levels`. One plot is produced for each |eta| and pt bin
    and for each method (fe, sm).
    The methods to take into account are given as `--categories`.
    """

    # how to create the branch map
    branching_type = "merged"

    # upstream workflow
    input_task_cls = AlphaExtrapolation

    # keys for looking up input task results
    input_keys = ("width", "extrapolation")

    # plot file names will start with this
    output_prefix = "extp"

    # plot configuration (e.g. axes limits/labels/scales)
    plot_settings = {
        "legend_kwargs": dict(loc="lower right"),
        "xlabel": r"$\alpha_{max}$",
        "xlim": (0, 0.3),
        "xscale": "linear",
        "ylabel": "Asymmetry width",
        "ylim": (0, 0.25),
        "yscale": "linear",
    }

    # which variable keys to use for constructing branch map
    branch_map_binning_variable_keys = ("abseta",)  # TODO: add pt

    #
    # helper methods for handling task inputs/outputs
    #

    def load_input(self, key: str, sample: str, level: str):
        input_collection = self.input()["input_task"]["collection"]
        coll_keys = [
            coll_key
            for coll_key, coll in input_collection.targets.items()
            if sample in coll[key]
        ]
        if len(coll_keys) != 1:
            raise RuntimeError(
                f"found {len(coll_keys)} input collections corresponding to "
                f"sample '{sample}', expected 1",
            )
        return input_collection[coll_keys[0]][key][sample][level].load(formatter="pickle")

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
        variable_map = self.postprocessor_inst.variable_map

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
        ref_object = next(_iter_flat(inputs["width"]))

        # a mapping of variable keys to lists o
        # dicts, each containing information about one bin
        # in the corresponding variable (excluding variables
        # that are already part of the branch map)
        bv_var_keys = set(variable_map["reco"]) - set(self.branch_map_binning_variable_keys) - {"asymmetry", "alpha"}
        bv_bin_lists: dict[list[dict]] = self._get_filtered_variable_bins(
            variable_map["reco"],
            bv_var_keys,
            from_hist=ref_object,
            remove_skipped=False,
        )

        # loop through bins and do plotting
        plt.style.use(mplhep.style.CMS)
        for bv_bins in product_dict({
            "category": self.config_inst.x.method_categories,
            **bv_bin_lists,
        }):
            # pop category and retrieve config object
            category = bv_bins.pop("category")
            category_inst = self.config_inst.get_category(category)

            # skip bin if requested
            if any(
                bv_bin.get("skip", False)
                for bv_bin in bv_bins.values()
            ):
                bin_slug = "/".join(
                    slug for bv_bin in bv_bins.values()
                    if (slug := bv_bin.get("slug", None))
                )
                logger.warning_once(f"skipping bin: {bin_slug}")
                continue

            # initialize figure and axes
            fig, ax = plt.subplots()
            mplhep.cms.label(
                lumi=round(0.001 * self.config_inst.x.luminosity.get("nominal"), 2),  # /pb -> /fb
                com=f"{self.config_inst.campaign.ecm:g}",
                ax=ax,
                llabel="Private Work",
                data=True,
            )

            # construct selectors for slicing histogram to get current bin
            bin_selectors = {}
            for level in self.levels:
                bin_selectors[level] = bin_selectors_level = {
                    "category": hist.loc(category),
                }
                # selectors that are part of branch map
                for bv_key in self.branch_map_binning_variable_keys:
                    bin_selectors_level[variable_map[level][bv_key]] = hist.loc(
                        self.branch_data[bv_key]["loc"],
                    )

                # selectors that are part of inner loop
                for bv_key, bv_bin in bv_bins.items():
                    bin_selectors_level[variable_map[level][bv_key]] = (
                        hist.loc(category_inst.name)
                        if bv_key == "category"
                        else hist.loc(bv_bin["loc"])
                    )

            # loop through samples/levels
            for sample, level in itertools.product(self.samples, self.levels):
                # only use reco level for data
                if sample == "data" and level != "reco":
                    continue

                # get input histogram for widths
                h_in = inputs["width"][sample][level]
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
                        "marker": "d",
                    })

                # plot widths
                alpha_edges = h_sliced.axes[variable_map[level]["alpha"]].edges
                plot_xy(
                    alpha_edges[1:],  # use upper edge to set point
                    h_sliced.values(),
                    yerr=np.sqrt(h_sliced.variances()),
                    ax=ax,
                    **plot_kwargs,
                )

                # get extrapolation fit results
                h_inter = inputs["extrapolation"][sample][level]["intercepts"]
                h_slope = inputs["extrapolation"][sample][level]["slopes"]
                h_inter_sliced = h_inter[bin_selectors[level]]
                h_slope_sliced = h_slope[bin_selectors[level]]

                # compute plot points for extrapolation function
                extp_x = np.linspace(alpha_edges[0], alpha_edges[-1], 100)
                extp_y = h_inter_sliced.value + h_slope_sliced.value * extp_x

                # plot extrapolation function
                ax.plot(
                    extp_x, extp_y,
                    color=plot_kwargs.get("color", None),
                    linestyle="dashed" if level == "gen" else "solid",
                )

            #
            # annotations
            #

            # texts to display on plot
            annotation_texts = [
                # category label
                category_inst.label,
            ] + [
                self.branch_data[bv_key]["label"]
                for bv_key in self.branch_map_binning_variable_keys
            ] + [
                bv_bin["label"]
                for bv_bin in bv_bins.values()
            ]

            # curry function for convenience
            annotate = partial(annotate_corner, ax=ax, loc="upper left")

            # add annotations to plot
            for i, text in enumerate(annotation_texts):
                annotate(
                    text=text,
                    xy_offset=(20, -20 - 30 * i),
                )

            # figure adjustments
            self.apply_plot_settings(ax=ax, context={"vars": variable_map})

            # legend
            ax.legend(**self.plot_settings.get("legend_kwargs", {}))

            # compute plot filename
            fname_parts = [
                # base name
                self.output_prefix,
                # method
                category,
            ] + [
                self.branch_data[bv_key]["slug"]
                for bv_key in self.branch_map_binning_variable_keys
            ] + [
                bv_bin["slug"]
                for bv_bin in bv_bins.values()
            ]

            # save plot to file
            self.save_plot("__".join(fname_parts), fig)
            plt.close(fig)
