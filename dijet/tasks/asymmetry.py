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
from dijet.util import deflate_dict

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

    # post-processor steps
    postprocessor_steps = ("trim_tails",)

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
            # add top-level sample key (for easier output lookup by plotting task)
            self.branch_data.sample: {
                # key indicating result produced by processing step
                output_key: {
                    # level: 'gen' (MC-only) or 'reco'
                    level: self.target(f"{'__'.join(output_key.split('.'))}__{level}.pickle")
                    for level in self.valid_levels
                }
                for output_key in self.output_keys
            },
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

    @property
    def input_keys(self):
        """
        Get input keys from first post-processor step
        """
        if not self.postprocessor_steps:
            return set()
        step = self.postprocessor_steps[0]
        return set(self.postprocessor_inst.steps[step]["inputs"])

    @property
    def output_keys(self):
        """
        Collect output keys from all postprocessor steps.
        """
        output_keys = set()
        for step in self.postprocessor_steps:
            output_keys |= set(self.postprocessor_inst.steps[step]["outputs"])
        return output_keys

    def load_histogram(self, dataset: str, variable: str):
        """
        Load histogram for a single `dataset` and `variable`
        from `MergeHistograms` outputs.
        """
        histogram = self.input()[dataset]["collection"][0]["hists"].targets[variable].load(formatter="pickle")
        return histogram

    def dump_output(self, output_key: str, level: str, obj: object):
        """
        Helper function for writing output to pickle file at the appropriate path.
        """

        # details of path in nested output dict
        path = [
            ("sample", self.branch_data.sample),
            ("output_key", output_key),
            ("level", level),
        ]

        # iteratively get target from nested output dict
        output = self.output()
        for key_label, key in path:
            try:
                output = output[key]
            except KeyError:
                raise KeyError(
                    f"no output registered for {key_label} '{key}'",
                )

        # write the output
        output.dump(obj, formatter="pickle")

    #
    # task implementation
    #

    def _run_impl(self, datasets: list[od.Dataset], level: str):
        """
        Implementation of asymmetry calculation.
        """
        # check provided level
        if level not in ("gen", "reco"):
            raise ValueError(f"invalid level '{level}', expected one of: gen,reco")

        # collect outputs here
        hists = {}

        # read histograms for input responses
        for response_key, response_cfg in self.postprocessor_inst.responses.items():
            # skip derived responses
            if response_cfg.get("derived", False):
                continue
            # retrieve histogram variable
            hist_var = response_cfg["hist_vars"][level]

            # load input hists
            h_all = []
            for dataset in datasets:
                h_in = self.load_histogram(dataset, hist_var)
                h_in_reduced = self.reduce_histogram(
                    h_in,
                    self.shift,
                    level,
                )
                h_all.append(h_in_reduced)

            # sum over all datasets
            h_sum = sum(h_all)

            # put inputs into structure expected by post-processor
            hists[response_key] = {
                "dist": h_sum,
            }

        # deflate nested input dict before passing
        # to postprocessor
        hists = deflate_dict(hists, max_depth=2)

        # run post-processing steps
        for step in self.postprocessor_steps:
            hists.update(self.postprocessor_inst.run_step(
                task=self,
                step=step,
                inputs=hists,
                level=level,
            ))

        # return
        return hists

    def run(self):
        # process histograms for all applicable levels
        sample = self.branch_data.sample
        for level in self.valid_levels:
            print(f"computing asymmetry for {sample = !r}, {level = !r}")
            results = self._run_impl(self.branch_data.datasets, level=level)

            print("writing outputs")
            for output_key in self.output_keys:
                result = results[output_key]
                self.dump_output(output_key, level, result)
