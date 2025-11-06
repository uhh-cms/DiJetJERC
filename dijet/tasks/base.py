# coding: utf-8

"""
Custom base tasks.
"""
from __future__ import annotations

import law

from functools import partial

from columnflow.tasks.framework.base import BaseTask, ShiftTask
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorMixin, ReducerMixin, ProducersMixin,
    HistProducerMixin, CategoriesMixin,
)
from columnflow.tasks.histograms import MergeHistograms
from columnflow.util import dev_sandbox, DotDict

from dijet.tasks.mixins import PostProcessorMixin, DiJetSamplesMixin
from dijet.util import get_variable_for_level


class DiJetTask(BaseTask):
    task_namespace = "dijet"


class HistogramsBaseTask(
    DiJetTask,
    CalibratorsMixin,
    SelectorMixin,
    ReducerMixin,
    ProducersMixin,
    HistProducerMixin,
    PostProcessorMixin,
    CategoriesMixin,
    DiJetSamplesMixin,
    ShiftTask,
):
    """
    Base task to load histogram and reduce them to used information.
    An example implementation of how to handle the inputs in a run method can be
    found in columnflow/tasks/histograms.py
    """
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # declare output as a nested sibling file collection
    output_collection_cls = law.NestedSiblingFileCollection
    output_base_keys = ()
    resolution_task_cls = MergeHistograms
    single_config = True

    # ways of creating the branch map
    branching_types = ("separate", "with_mc", "merged")
    branching_type = None  # set in derived tasks

    #
    # law parameters
    #

    levels = law.CSVParameter(
        default=("reco", "gen"),
        description="comma-separated list of 'reco', 'gen', or both, indicating whether to "
        "use the regular (reconstruction-level) variables or the equivalent variable on "
        "generator-level; if not given, only 'reco'-level variables are used",
        choices={"reco", "gen"},
        brace_expand=True,
        parse_empty=True,
    )

    #
    # class methods
    #

    @property
    def valid_levels(self):
        """List of levels that are valid for the branch task (i.e. no 'gen' level for data)."""
        # no unique sample information if called in a workflow context
        # -> return list of levels directly
        if self.is_workflow():
            return self.levels
        else:
            return [level for level in self.levels if self.branch_data.is_mc or level != "gen"]

    def iter_levels_variables(self, levels: list[str] | None = None):
        """
        Generator yielding tuples of the form (level, variable), with *level*
        being either 'reco' for reconstruction-level or 'gen' for gen-level
        variables. An *levels* argument can be provided to restrict the levels
        (e.g. gen-level in MC).
        """
        levels_ = levels or self.valid_levels
        yield from super().iter_levels_variables(levels=levels_)

    def create_branch_map(self):
        """
        Branching map for workflow. Depends on the *branching_type* class
        variable: if *separate*, the workflow will have one branch for each
        sample, if *merged*, there will be a single branch covering all samples.
        """
        # check branching type
        if self.branching_type is None:
            branching_types_str = ",".join(self.branching_types)
            raise ValueError(
                f"missing branching type for task '{self.task_family}', "
                f"derived tasks must implement `branching_type` class "
                f"variable; valid choices are: {branching_types_str}",
            )
        elif self.branching_type not in self.branching_types:
            branching_types_str = ",".join(self.branching_types)
            raise ValueError(
                f"invalid branching type '{self.branching_type}', "
                f"valid choices are: {branching_types_str}",
            )

        branches = []
        for sample in sorted(self.samples):
            datasets = self.get_datasets(self.config_inst, [sample])

            # check if datasets in the sample are MC or data and set flag
            sample_is_mc = set(
                self.config_inst.get_dataset(d).is_mc
                for d in datasets
            )

            # raise exception if sample contains both data and MC
            if len(sample_is_mc) > 1:
                datasets_str = ",".join(datasets)
                raise RuntimeError(
                    f"datasets for sample `{sample}` have mismatched "
                    f"`is_mc` flags: {datasets_str}",
                )

            branches.append(DotDict.wrap({
                "sample": sample,
                "datasets": datasets,
                "is_mc": list(sample_is_mc)[0],
            }))

        if self.branching_type == "separate":
            # return branch map directly
            return branches

        elif self.branching_type == "with_mc":
            # like 'separate', but check that there is exactly one
            # MC dataset and add it to the branch data
            mc_branches = [b for b in branches if b.is_mc]
            if len(mc_branches) != 1:
                raise ValueError(
                    f"expected exactly 1 MC branch, got {len(mc_branches)}",
                )

            for b in branches:
                b["mc_sample"] = mc_branches[0]["sample"]

            return branches

        elif self.branching_type == "merged":
            # return only one branch representing the merging of exactly
            # one data and one MC sample
            data_branches = [b for b in branches if not b.is_mc]
            if len(data_branches) != 1:
                raise ValueError(
                    f"expected exactly 1 data branch, got {len(data_branches)}",
                )

            mc_branches = [b for b in branches if b.is_mc]
            if len(mc_branches) != 1:
                raise ValueError(
                    f"expected exactly 1 MC branch, got {len(mc_branches)}",
                )

            return [
                DotDict.wrap({
                    "data_sample": data_branches[0]["sample"],
                    "mc_sample": mc_branches[0]["sample"],
                    "datasets": data_branches[0]["datasets"] + mc_branches[0]["datasets"],
                }),
            ]

        else:
            raise NotImplementedError(
                f"internal error: branching type {self.branching_type} not implemented",
            )

        return branches

    def store_parts(self) -> law.util.InsertableDict[str, str]:
        """
        Add dijet-specific parts to the output path:
        - *sample*: the sample being processed; if *branching_type* is
          'merged', this will be a representation of all sample names
        """
        # FIXME: ideally move to `DiJetSamplesMixin`, but doesn't seen to work atm due to mro complications
        parts = super().store_parts()

        # insert 'sample' part after 'dataset'
        if self.branching_type in ("separate", "with_mc"):
            parts.insert_after(
                "dataset",
                "sample",
                f"{self.branch_data.sample}",
            )

        elif self.branching_type == "merged":
            parts.insert_after(
                "dataset",
                "sample",
                f"{self.branch_data.data_sample}__{self.branch_data.mc_sample}",
            )

        else:
            assert False, "internal error"

        # remove 'dataset' part from store_parts
        parts.pop("dataset", None)

        return parts

    def reduce_histogram(self, histogram, shift, level):
        """
        Reduce away the `shift` and `process` axes of a multidimensional
        histogram by selecting a single shift and summing over all processes.
        """
        import hist

        def flatten_nested_list(nested_list):
            return [item for sublist in nested_list for item in sublist]

        # get shift instance
        shift_inst = self.config_inst.get_shift(shift)
        if shift_inst.name not in histogram.axes["shift"]:
            raise ValueError(f"histogram does not contain shift `{shift}`")

        # work on a copy
        h = histogram.copy()

        # axis reductions
        h = h[{
            "process": sum,
            "shift": hist.loc(shift_inst.name),
            # TODO: read rebinning factors from config
            # @dsavoiu: might be better to use config binning for now
            # vars_["alpha"]: hist.rebin(5),
            # vars_["asymmetry"]: hist.rebin(2),
        }]

        return h
