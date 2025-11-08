# coding: utf-8
"""
Task for plotting JER scale factors as a function
of pt.
"""
from __future__ import annotations

import law

from functools import partial

from columnflow.util import maybe_import, load_correction_set
from columnflow.tasks.external import BundleExternalFiles

from dijet.tasks.sf import SF
from dijet.plotting.base import PlottingBaseTask
from dijet.plotting.util import annotate_corner, plot_xy
from dijet.util import product_dict, inflate_dict

logger = law.logger.get_logger(__name__)

hist = maybe_import("hist")
np = maybe_import("numpy")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")


class PlotSF(
    PlottingBaseTask,
):
    """
    Task to plot the JER scale factors.

    Shows the ratios of JERs extracted from data and MC
    provided via `--samples`. One plot is produced for each |eta| bin
    for each method (fe, sm).
    The methods to take into account are given as `--categories`.
    """

    # how to create the branch map
    branching_type = "merged"

    # upstream workflow
    input_task_cls = SF

    # plot file names will start with this
    output_prefix = "sf"

    # plot configuration (e.g. axes limits/labels/scales)
    plot_settings = {
        "legend_kwargs": dict(loc="upper right"),
        "xlabel": lambda self, ctx: self.config_inst.get_variable(ctx["vars"]["pt"]).get_full_x_title(),  # noqa
        "xlim": (49, 2100),
        "xscale": "log",
        "ylabel": "Jet energy resolution scale factor",
        "ylim": (0.5, 3.5),
        "yscale": "linear",
    }

    # which variable keys to use for constructing branch map
    branch_map_binning_variable_keys = ("abseta",)

    #
    # methods required by law
    #

    def requires(self):
        """
        Include external files bundle in requirements (for plotting JER SF
        from correctionlib files)
        """
        deps = super().requires()

        # add external files requirement
        deps["external_files"] = BundleExternalFiles.req(self)

        return deps

    #
    # helper methods for handling task inputs/outputs
    #

    def load_input(self, input_key: str):
        """
        Load a single input from a pickle file.
        """
        # map of input task branch to collection of output targets
        input_collection = self.input()["input_task"]["collection"]
        coll_keys = [
            coll_key
            for coll_key, coll in input_collection.targets.items()
        ]
        if len(coll_keys) != 1:
            raise RuntimeError(
                f"found {len(coll_keys)} input collections, expected 1",
            )

        # retrieve target for input key
        input_target = input_collection[coll_keys[0]][input_key]

        # load pickle file
        return input_target.load(formatter="pickle")

    def load_inputs(self):
        return {
            input_key: self.load_input(input_key)
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
        variable_map = self.postprocessor_inst.variable_map["reco"]

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

        # load correction object for official JER SF
        # TODO: make optional
        jer_cfg = self.config_inst.x.jer["Jet"]
        jer_sf_key = f"{jer_cfg.campaign}_{jer_cfg.version}_MC_ScaleFactor_{jer_cfg.jet_type}"
        correction_set = load_correction_set(self.requires()["external_files"].files["jet_jerc"])
        correction = correction_set[jer_sf_key]

        # binning information from first histogram object
        # (assume identical binning for all)
        def _iter_flat(d: dict):
            if not isinstance(d, dict):
                yield d
                return
            for k, v in d.items():
                yield from _iter_flat(v)

        # retrieve response configuration
        response_key = self.postprocessor_inst.calc_jer_main_response
        response_cfg = self.postprocessor_inst.responses[response_key]

        # get binning information from first histogram object
        # (assume identical binning for all)
        ref_object = next(_iter_flat(inputs[response_key]["jer_sf"]))

        # variable key corresponsing to response distribution
        response_var_key = response_cfg["response_var_key"]

        # a mapping of variable keys to lists of
        # dicts, each containing information about one bin
        # in the corresponding variable (excluding variables
        # that are already part of the branch map)
        bv_var_keys = (
            set(response_cfg["all_var_keys"]) -
            set(self.branch_map_binning_variable_keys) -
            {self.postprocessor_inst.extrapolation_var_key} -
            {response_var_key} -
            {"pt"}
        )
        bv_bin_lists: dict[list[dict]] = self._get_filtered_variable_bins(
            variable_map,
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
            bin_selector = {
                "category": hist.loc(category),
            }
            # selectors that are part of branch map
            for bv_key in self.branch_map_binning_variable_keys:
                bin_selector[variable_map[bv_key]] = hist.loc(
                    self.branch_data[bv_key]["loc"],
                )

            # selectors that are part of inner loop
            for bv_key, bv_bin in bv_bins.items():
                bin_selector[variable_map[bv_key]] = (
                    hist.loc(category_inst.name)
                    if bv_key == "category"
                    else hist.loc(bv_bin["loc"])
                )

            # get input histogram for widths
            h_in = inputs[response_key]["jer_sf"]
            h_sliced = h_in[bin_selector]

            # plot asymmetry distribution
            plot_kwargs = dict(
                self.config_inst.x("samples", {})
                .get("data", {}).get("plot_kwargs", {}),
            )
            # resolve task-specific kwargs
            plot_kwargs = plot_kwargs.get(
                self.__class__,
                plot_kwargs.get("__default__", {}),
            )

            # compute legend entry
            def get_sample_label(sample):
                return (
                    self.config_inst.x("samples", {})
                    .get(sample, {}).get("label", sample)
                )
            plot_kwargs["label"] = "/".join([
                get_sample_label(self.branch_data.data_sample),
                get_sample_label(self.branch_data.mc_sample),
            ])

            # plot JER SF
            plot_xy(
                h_sliced.axes[variable_map["pt"]].centers,
                h_sliced.values(),
                yerr=np.sqrt(h_sliced.variances()),
                ax=ax,
                **plot_kwargs,
            )

            # plot official JER SF values from correction object
            # TODO: add support for pT-dependent JER SFs
            ax.axhline(
                correction.evaluate(self.branch_data.abseta.loc, "nom"),
                color="gray",
                linestyle="dashed",
                linewidth=2,
                label=f"{jer_cfg.campaign}_{jer_cfg.version}",
                zorder=-10,
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
                # output prefix from task
                self.output_prefix,
                # name of response
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
