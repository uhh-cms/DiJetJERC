# coding: utf-8

"""
Custom base tasks.
"""

import law

from functools import partial

from columnflow.tasks.framework.base import BaseTask, ShiftTask
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorMixin, ProducersMixin,
    CategoriesMixin,
)
from columnflow.config_util import get_datasets_from_process
from columnflow.util import dev_sandbox, DotDict

from dijet.tasks.mixins import DiJetVariablesMixin, DiJetSamplesMixin


class DiJetTask(BaseTask):
    task_namespace = "dijet"


class HistogramsBaseTask(
    DiJetTask,
    DiJetSamplesMixin,
    CategoriesMixin,
    DiJetVariablesMixin,
    ProducersMixin,
    SelectorMixin,
    CalibratorsMixin,
    ShiftTask,
):
    """
    Base task to load histogram and reduce them to used information.
    An example implementation of how to handle the inputs in a run method can be
    found in columnflow/tasks/histograms.py
    """
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # Add nested sibling directories to output path
    output_collection_cls = law.NestedSiblingFileCollection
    output_per_level = True  # if True, declared output paths will not depend on level
    output_base_keys = ()

    # ways of creating the branch map
    branching_types = ("separate", "merged")
    branching_type = None  # set in derived tasks

    # Category ID for methods
    LOOKUP_CATEGORY_ID = {"sm": 1, "fe": 2}

    @staticmethod
    def _io_key(base_key, level: str):
        return base_key if level == "reco" else f"{base_key}_{level}"

    def single_input(self, base_key, level: str):
        return self.input()[self._io_key(base_key, level)]

    def single_output(self, base_key, level: str):
        if self.output_per_level:
            return self.output()[self._io_key(base_key, level)]
        else:
            # TODO: refactor output as dict with levels as keys
            return self.output()[base_key]

    def output(self) -> dict[law.FileSystemTarget]:
        # declare output files
        outp = {}
        output_levels = ["reco"] if not self.output_per_level else self.levels
        for level in output_levels:
            # skip gen level in data
            if not self.branch_data.is_mc and level == "gen":
                continue

            # register output files for level
            for basename in self.output_base_keys:
                key = self._io_key(basename, level) if self.output_per_level else basename  # noqa
                outp[key] = self.target(f"{key}.pickle", type="f")

        return outp

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
                    "`is_mc` flags: {datasets_str}",
                )

            branches.append(DotDict.wrap({
                "sample": sample,
                "datasets": datasets,
                "is_mc": list(sample_is_mc)[0],
            }))

        if self.branching_type == "separate":
            # return branch map directly
            return branches

        elif self.branching_type == "merged":
            # return only data branch
            # TODO: don't hardcode data sample name
            # TODO: 'samples' instead of 'sample' in the branch data
            return [b for b in branches if b.sample == "data"]

        else:
            assert False, "internal error"

        return branches

    def store_parts(self) -> law.util.InsertableDict[str, str]:
        """
        Add dijet-specific parts to the output path:
        - *sample*: the sample being processed; if *branching_type* is
          'merged', this will be a representation of all sample names
        """
        parts = super().store_parts()

        if self.branching_type == "separate":
            parts.insert_after("version", "sample", f"{self.branch_data.sample}")

        elif self.branching_type == "merged":
            parts.insert_after("version", "sample", f"{self.samples_repr}")

        else:
            assert False, "internal error"

        return parts

    def reduce_histogram(self, histogram, shift, level):
        """
        Reduce away the `shift` and `process` axes of a multidimensional
        histogram by selecting a single shift and summing over all processes.
        """
        import hist

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

        def flatten_nested_list(nested_list):
            return [item for sublist in nested_list for item in sublist]

        # get shift instance
        shift_inst = self.config_inst.get_shift(shift)
        if shift_inst.id not in histogram.axes["shift"]:
            raise ValueError(f"histogram does not contain shift `{shift}`")

        # work on a copy
        h = histogram.copy()

        # axis reductions
        h = h[{
            "process": sum,
            "shift": hist.loc(shift_inst.id),
            # TODO: read rebinning factors from config
            # @dsavoiu: might be better to use config binning for now
            # vars_["alpha"]: hist.rebin(5),
            vars_["asymmetry"]: hist.rebin(2),
        }]

        return h
