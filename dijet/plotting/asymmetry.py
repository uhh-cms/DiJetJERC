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
from dijet.plotting.util import annotate_corner, plot_xy
from dijet.util import product_dict, inflate_dict, iter_flat_dict

logger = law.logger.get_logger(__name__)

hist = maybe_import("hist")
np = maybe_import("numpy")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")


class PlotAsymmetry(
    PlottingBaseTask,
):
    """
    Task to plot asymmetry distributions.

    Shows the asymmetry distribution for all given `--samples`
    and `--levels`. One plot is produced for each |eta|, pt and alpha bin
    and for each method (fe, sm).
    The methods to take into account are given as `--categories`.
    """

    # how to create the branch map
    branching_type = "merged"

    # upstream workflow
    input_task_cls = Asymmetry

    # plot file names will start with this
    output_prefix = "dist"

    # plot configuration (e.g. axes limits/labels/scales)
    plot_settings = {
        "legend_kwargs": dict(loc="upper right"),
        "xlabel": lambda self, ctx: self.config_inst.get_variable(ctx["vars"]["reco"][ctx["response_var_key"]]).x_title,
        "xlim": (-0.5, 0.5),
        "xscale": "linear",
        "ylabel": r"$\Delta$N/N",
        "ylim": (5e-5, 10),
        "yscale": "log",
    }

    # which variable keys to use for constructing branch map
    branch_map_binning_variable_keys = ("abseta",)  # TODO: add alpha, pt

    #
    # helper methods for handling task inputs/outputs
    #

    def load_input(self, input_key: str, sample: str, level: str):
        """
        Load a single input from a pickle file.
        """
        # map of input task branch to collection of output targets
        input_collection = self.input()["input_task"]["collection"]

        # find branch for sample
        coll_keys = [
            coll_key
            for coll_key, coll in input_collection.targets.items()
            if sample in coll
        ]
        if len(coll_keys) != 1:
            raise RuntimeError(
                f"found {len(coll_keys)} input collections corresponding to "
                f"sample '{sample}', expected 1",
            )

        # retrieve target for input key
        input_target = input_collection[coll_keys[0]][sample][input_key][level]

        # load pickle file
        return input_target.load(formatter="pickle")

    def load_inputs(self):
        """
        Load all inputs and return them in a structured dictionary.
        """
        return {
            input_key: {
                sample: {
                    level: self.load_input(input_key, sample=sample, level=level)
                    for level in self.levels
                    if level == "reco" or sample != "data"
                }
                for sample in self.samples
            }
            for input_key in self.input_keys
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

        # helper function for input preparation (clean nans)
        def _prepare_input(histogram):
            # map `nan` values to zero
            v = histogram.view()
            v.value = np.nan_to_num(v.value, nan=0.0)
            v.variance = np.nan_to_num(v.variance, nan=0.0)
            # return prepared histogram
            return histogram

        # iteratively apply preparation to nested input dict
        inputs = law.util.map_struct(_prepare_input, raw_inputs)

        # expand dot-separated keys to multi-level
        inputs = inflate_dict(inputs)

        # go through all response variables and produce plots
        for response_key, response_cfg in self.postprocessor_inst.responses.items():
            # skip derived responses (not yet computed at this stage)
            if response_cfg.get("derived", False):
                continue

            # get binning information from first histogram object
            # (assume identical binning for all)
            ref_object = next(iter_flat_dict(inputs[response_key]["norm"]))

            # variable key corresponsing to response distribution
            response_var_key = response_cfg["response_var_key"]

            # a mapping of variable keys to lists of
            # dicts, each containing information about one bin
            # in the corresponding variable (excluding variables
            # that are already part of the branch map)
            bv_var_keys = (
                set(response_cfg["all_var_keys"]) -
                set(self.branch_map_binning_variable_keys) -
                {response_var_key}
            )
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

                    # get input histogram for asymmetries
                    h_in = inputs[response_key]["norm"][sample][level]
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
                        h_sliced.axes[variable_map[level][response_var_key]].centers,
                        h_sliced.values(),
                        xerr=h_sliced.axes[variable_map[level][response_var_key]].widths / 2,
                        yerr=np.sqrt(h_sliced.variances()),
                        ax=ax,
                        **plot_kwargs,
                    )

                    # plot quantiles as vertical lines
                    hs_in_quantiles = inputs[response_key]["cut_edges"][sample][level]
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
                self.apply_plot_settings(ax=ax, context={"vars": variable_map, "response_var_key": response_var_key})

                # legend
                ax.legend(**self.plot_settings.get("legend_kwargs", {}))

                # compute plot filename
                fname_parts = [
                    # output prefix
                    self.output_prefix,
                    # base name is histogram name
                    response_key,
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
