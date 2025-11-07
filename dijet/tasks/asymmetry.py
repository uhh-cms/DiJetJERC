# coding: utf-8

"""
Custom tasks to derive JER SF.
"""

import law
import order as od

from law.util import flatten

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

    # how to create the branch map
    branching_type = "separate"

    # upstream requirements
    reqs = Requirements(
        MergeHistograms=MergeHistograms,
    )

    @property
    def output_base_keys(self):
        return self.postprocessor_inst.steps["trim_tails"].outputs

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

            # arbitrary struct of multidimensional variables,
            # organized by level and histogram key
            variables_for_histograms = self.postprocessor_inst.variables_func(self)

            # get variables as flat list and pass on to dependent task
            # (including gen-level variables only for MC)
            variables = []
            for level in self.levels:
                if level == "gen" and not dataset_inst.is_mc:
                    continue
                variables.extend(flatten(variables_for_histograms[level]))

            # register `MergeHistograms` as requirement,
            # setting `variables` by hand
            reqs[dataset] = self.reqs.MergeHistograms.req_different_branching(
                self,
                dataset=dataset,
                branch=-1,
                variables=variables,
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

    def _run_impl(self, datasets: list[od.Dataset], level: str, variables: dict[str]):
        """
        Implementation of asymmetry calculation.
        """
        # check provided level
        if level not in ("gen", "reco"):
            raise ValueError(f"invalid level '{level}', expected one of: gen,reco")

        # helper function for loading hists and summing over all datasets
        def prepare_hist(variable):
            h_all = []
            for dataset in datasets:
                h_in = self.load_histogram(dataset, variable)
                h_in_reduced = self.reduce_histogram(
                    h_in,
                    self.shift,
                    level,
                )
                h_all.append(h_in_reduced)
            return sum(h_all)

        # put inputs into structure expected by post-processor
        hists = {
            hist_key: prepare_hist(var_name)
            for hist_key, var_name in variables.items()
        }

        # run post-processing step for computing trimmed asymmetry
        hists.update(self.postprocessor_inst.run_step(
            task=self,
            step="trim_tails",
            inputs=hists,
            level=level,
        ))

        # store outputs in pickle file for further processing
        for key in self.output_base_keys:
            self.dump_output(key, level=level, obj=hists[key])

    def run(self):
        # process histograms for all applicable levels
        sample = self.branch_data.sample
        for level, variables in self.iter_levels_histogram_variables():
            variables_str = ", ".join(variables.values())
            print(f"computing asymmetries for {sample = !r}, {level = !r}, variables = {variables_str!r}")
            self._run_impl(self.branch_data.datasets, level=level, variables=variables)
