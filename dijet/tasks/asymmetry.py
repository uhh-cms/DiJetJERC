# coding: utf-8

"""
Custom tasks to derive JER SF.
"""

import law
import order as od

from functools import partial

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.histograms import MergeHistograms
from columnflow.util import maybe_import

from dijet.tasks.base import HistogramsBaseTask

ak = maybe_import("awkward")
hist = maybe_import("hist")
np = maybe_import("numpy")


class Asymmetry(
    HistogramsBaseTask,
):
    """
    Task to prepare asymmetries for width extrapolation.
    Read in and plot asymmetry histograms.
    Cut off non-gaussian tails.
    """

    # Add nested sibling directories to output path
    output_collection_cls = law.NestedSiblingFileCollection

    # upstream requirements
    reqs = Requirements(
        MergeHistograms=MergeHistograms,
    )

    def requires(self):
        reqs = {}

        for dataset in self.datasets:
            # get dataset instance
            dataset_inst = self.config_inst.get_dataset(dataset)

            # variables to pass on to dependent task (including
            # gen-level variables only for MC)
            variables_for_histograms = [
                variable
                for level, variable in self.iter_levels_variables()
                if level == "reco" or dataset_inst.is_mc
            ]

            # register `MergeHistograms` as requirement,
            # setting `variables` by hand
            reqs[dataset] = self.reqs.MergeHistograms.req(
                self,
                dataset=dataset,
                branch=-1,
                variables=variables_for_histograms,
                _exclude={"branches"},
            )

        return reqs

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["merged_hists"] = self.requires_from_branch()
        return reqs

    def load_histogram(self, dataset, variable):
        histogram = self.input()[dataset]["collection"][0]["hists"].targets[variable].load(formatter="pickle")
        return histogram

    def output_key(self, base_key, level: str):
        return base_key if level == "reco" else f"{base_key}_{level}"

    def output(self) -> dict[law.FileSystemTarget]:
        # TODO: Unstable for changes like data_jetmet_X
        #       Make independent like in config datasetname groups

        # set target directory
        sample = self.extract_sample()
        target = self.target(f"{sample}", dir=True)

        # check if MC or data
        _, isMC = self.get_datasets()

        # declare output files
        outp = {}
        for level in self.levels:
            # skip gen level in data
            if not isMC and level == "gen":
                continue

            # register output files for level
            for basename in ("asym", "asym_cut", "nevt", "quantile"):
                key = self.output_key(basename, level)
                outp[key] = target.child(f"{key}.pickle", type="f")

        return outp

    def _run_impl(self, datasets: list[od.Dataset], level: str, variable: str):
        """
        Implementation of asymmetry calculation.
        """
        # check provided level
        if level not in ("gen", "reco"):
            raise ValueError(f"invalid level '{level}', expected one of: gen,reco")

        # suffix to use when looking up output files
        output_suffix = "" if level == "reco" else f"_{level}"

        # dict storing either variables or their gen-level equivalents
        # for convenient access
        resolve_var = partial(
            self._get_variable_for_level,
            config=self.config_inst,
            level=level,
        )
        vars_ = {
            "alpha": resolve_var(name=self.alpha_variable),
            "asymmetry": resolve_var(name=self.asymmetry_variable),
        }

        # load hists and sum over all datasets
        h_all = []
        for dataset in datasets:
            h_in = self.load_histogram(dataset, variable)
            h_in_reduced = self.reduce_histogram(
                h_in,
                self.processes,
                self.shift,
                level,
            )
            h_all.append(h_in_reduced)
        h_all = sum(h_all)

        axes_names = [a.name for a in h_all.axes]
        axes_indices = {
            var_key: axes_names.index(var_name)
            for var_key, var_name in vars_.items()
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
        self.output()[self.output_key("nevt", level)].dump(h_nevts, formatter="pickle")

        # normalize histogram to integral over asymmetry
        view.value = view.value / integral
        view.variance = view.variance / integral**2

        # Store asymmetries with gausstails for plotting
        # TODO: h_all is further adjusted in scope of this task
        #       retrospectivley changing this output as well?
        self.output()[self.output_key("asym", level)].dump(h_all, formatter="pickle")

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
        self.output()[self.output_key("quantile", level)].dump(h_quantiles, formatter="pickle")

        # Create mask to filter data; Only bins above/below qunatile bins
        asym_centers = h_all.axes[vars_["asymmetry"]].centers  # Use centers to keep dim of view.value
        asym_centers_reshaped = asym_centers.reshape(1, 1, 1, 1, -1)  # TODO: not hard-coded
        mask = (asym_centers_reshaped > asym_edges_lo) & (asym_centers_reshaped < asym_edges_up)

        # Filter non gaussian tailes
        view.value = np.where(mask, view.value, np.nan)
        view.variance = np.where(mask, view.variance, np.nan)

        # Store in pickle file for plotting task
        self.output()[self.output_key("asym_cut", level)].dump(h_all, formatter="pickle")


    def run(self):
        # TODO: Gen level for MC
        #       Correlated fit (in jupyter)

        datasets, isMC = self.get_datasets()

        # process histograms for all applicable levels
        for level, variable in self.iter_levels_variables():
            if level == "gen" and not isMC:
                continue
            self._run_impl(datasets, level=level, variable=variable)
