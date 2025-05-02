# coding: utf-8

"""
Custom tasks to derive JER SF.
"""

import law
import order as od

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.histograms import MergeHistograms
from columnflow.util import maybe_import

from dijet.tasks.base import HistogramsBaseTask

ak = maybe_import("awkward")
hist = maybe_import("hist")
np = maybe_import("numpy")


class Asymmetry(
    HistogramsBaseTask,
    law.LocalWorkflow,
):
    """
    Task to prepare asymmetries for width extrapolation.

    Processing steps:
    - read in multidimensional histograms containing asymmetry distributions
    - compute quantiles and cut off non-gaussian tails
    - compute number of events per bin
    """

    # declare output collection type and keys
    output_collection_cls = law.NestedSiblingFileCollection
    output_base_keys = ("asym", "asym_cut", "nevt", "quantile")

    # how to create the branch map
    branching_type = "separate"

    # upstream requirements
    reqs = Requirements(
        MergeHistograms=MergeHistograms,
    )

    #
    # methods required by law
    #

    def output(self):
        """
        Organize output as a (nested) dictionary. Output files will be in a single
        directory, which is determined by `store_parts`.
        """
        return {
            key: {
                self.branch_data.sample: {
                    level: self.target(f"{key}_{level}.pickle")
                    for level in self.valid_levels
                },
            }
            for key in self.output_base_keys
        }

    def requires(self):
        reqs = {}

        for dataset in self.datasets:
            # get dataset instance
            dataset_inst = self.config_inst.get_dataset(dataset)
            valid_levels = ["reco"]
            if dataset_inst.is_mc:
                valid_levels.append("gen")

            # variables to pass on to dependent task (including
            # gen-level variables only for MC)
            variables_for_histograms = [
                variable
                for level, variable in self.iter_levels_variables(
                    levels=valid_levels,
                )
            ]

            # register `MergeHistograms` as requirement,
            # setting `variables` by hand
            reqs[dataset] = self.reqs.MergeHistograms.req_different_branching(
                self,
                dataset=dataset,
                branch=-1,
                variables=variables_for_histograms,
            )

        return reqs

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["merged_hists"] = self.requires_from_branch()
        return reqs

    #
    # helper methods for handling task inputs/outputs
    #

    def load_histogram(self, dataset: str, variable: str):
        """
        Load histogram for a single `dataset` and `variable`
        from `MergeHistograms` outputs.
        """
        histogram = self.input()[dataset]["collection"][0]["hists"].targets[variable].load(formatter="pickle")
        return histogram

    def dump_output(self, key: str, level: str, obj: object):
        if key not in self.output_base_keys:
            raise ValueError(
                f"output key '{key}' not registered in "
                f"`{self.task_family}.output_base_keys`",
            )
        self.output()[key][self.branch_data.sample][level].dump(obj, formatter="pickle")

    #
    # task implementation
    #

    def _run_impl(self, datasets: list[od.Dataset], level: str, variable: str):
        """
        Implementation of asymmetry calculation.
        """
        # check provided level
        if level not in ("gen", "reco"):
            raise ValueError(f"invalid level '{level}', expected one of: gen,reco")

        # dict storing either variables or their gen-level equivalents
        # for convenient access
        vars_ = self._make_var_lookup(level=level)

        #
        # start main processing
        #

        # load hists and sum over all datasets
        h_all = []
        for dataset in datasets:
            h_in = self.load_histogram(dataset, variable)
            h_in_reduced = self.reduce_histogram(
                h_in,
                self.shift,
                level,
            )
            h_all.append(h_in_reduced)
        h_all = sum(h_all)

        axes_names = [a.name for a in h_all.axes]
        axes_indices = {
            var_key: axes_names.index(vars_[var_key])
            for var_key in ("alpha", "asymmetry")
        }
        view = h_all.view()

        # replace histogram contents with cumulative sum over alpha bins
        view.value = np.apply_along_axis(np.cumsum, axis=axes_indices["alpha"], arr=view.value)
        view.variance = np.apply_along_axis(np.cumsum, axis=axes_indices["alpha"], arr=view.variance)

        # Get integral of asymmetries as array
        # Skip over-/underflow bins (i.e. TH1F -> ComputeIntegral for UHH2)
        # h_all[{vars_["asymmetry"]: sum}] includes such bins
        integral = h_all.values().sum(axis=axes_indices["asymmetry"], keepdims=True)

        # Store for width extrapolation
        h_nevts = h_all.copy()
        h_nevts = h_nevts[{vars_["asymmetry"]: sum}]
        h_nevts.view().value = np.squeeze(integral)
        self.dump_output("nevt", level=level, obj=h_nevts)

        # normalize histogram to integral over asymmetry
        view.value = view.value / integral
        view.variance = view.variance / integral**2

        # Store asymmetries with gausstails for plotting
        # TODO: h_all is further adjusted in scope of this task
        #       retrospectivley changing this output as well?
        self.dump_output("asym", level=level, obj=h_all)

        # Cut off non gaussian tails using quantiles
        # NOTE: My first aim was to use np.quantile like
        #       quantiles_lo = np.quantile(view.value, 0.015, axis=-1, keepdim=True)
        #       This appraoch does not work with arrays containing histograms but only the raw data

        # Get relative contribution from each bin to overall number.
        # For each index, it adds the previous ones.
        # For a normalized hist, the last entry must be 1.
        percentage = np.cumsum(view.value, axis=-1)
        # TODO: add assert so the last element is always 1

        # TODO: As input parameter in task for uncertainties
        #       Also needed for task output path
        quantile_lo = 0.015  # 1.5 % / left tail
        quantile_up = 0.985  # 98.5 % / right tail

        # Find index of quantile
        # NOTE: No alternative found yet for apply_along_axis
        ind_lo = np.apply_along_axis(np.searchsorted, -1, percentage, quantile_lo, side="left")
        ind_up = np.apply_along_axis(np.searchsorted, -1, percentage, quantile_up, side="right")

        # Extend index array by one axis
        ind_lo = np.expand_dims(ind_lo, axis=-1)
        ind_up = np.expand_dims(ind_up, axis=-1)

        # Obtain bin edge value for qunatile index
        # Store in histogram structure
        # NOTE: Not sure to rebin a hsitogram from (:,:,80) to (:,:,1)
        #       Remove bin completly with sum, since only one value is needed
        asym_edges = h_all.axes[vars_["asymmetry"]].edges  # One dim more then view.value
        asym_edges_lo = asym_edges[ind_lo]  # Get value for lower quantile
        asym_edges_up = asym_edges[ind_up + 1]  # Store in histogram structure

        # Store in histogram strcuture for plotting task
        # For the mask in the next step we need the shape (:,:,:,1) and can't remove the asymmetry axis completely.
        h_asym_edges_lo = h_all.copy()[{vars_["asymmetry"]: sum}]
        h_asym_edges_up = h_asym_edges_lo.copy()
        h_asym_edges_lo.view().value = np.squeeze(asym_edges_lo)
        h_asym_edges_up.view().value = np.squeeze(asym_edges_up)
        h_quantiles = {
            "low": h_asym_edges_lo,
            "up": h_asym_edges_up,
        }
        self.dump_output("quantile", level=level, obj=h_quantiles)

        # Create mask to filter data; Only bins above/below qunatile bins
        asym_centers = h_all.axes[vars_["asymmetry"]].centers  # Use centers to keep dim of view.value
        asym_centers_reshaped = asym_centers.reshape(1, 1, 1, 1, -1)  # TODO: not hard-coded
        mask = (asym_centers_reshaped > asym_edges_lo) & (asym_centers_reshaped < asym_edges_up)

        # Filter non gaussian tailes
        view.value = np.where(mask, view.value, 0)
        view.variance = np.where(mask, view.variance, 0)

        # Store in pickle file for plotting task
        self.dump_output("asym_cut", level=level, obj=h_all)

    def run(self):
        # process histograms for all applicable levels
        sample = self.branch_data.sample
        for level, variable in self.iter_levels_variables():
            print(f"computing asymmetries for {sample = !r}, {level = !r}, {variable = !r}")
            self._run_impl(self.branch_data.datasets, level=level, variable=variable)
