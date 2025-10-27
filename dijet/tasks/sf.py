# coding: utf-8

"""
Custom tasks to derive JER SF.
"""

import law

from dijet.tasks.base import HistogramsBaseTask
from columnflow.util import maybe_import
from columnflow.tasks.framework.base import Requirements

from dijet.tasks.jer import JER

np = maybe_import("numpy")


class SF(
    HistogramsBaseTask,
    law.LocalWorkflow,
):
    """
    Task to derive the JER SFs using the JERs in data and MC as inputs.

    Requires a post-processor that implements the following steps:
    - ``calc_sf`` (inputs: ``jer_data``, ``jer_mc``, outputs: ``sf``)
    """

    # declare output collection type and keys
    output_collection_cls = law.NestedSiblingFileCollection

    # how to create the branch map
    branching_type = "merged"

    # upstream requirements
    reqs = Requirements(
        JER=JER,
    )

    @property
    def output_base_keys(self):
        return {
            output
            for step in ("calc_sf",)
            for output in self.postprocessor_inst.steps[step].outputs
        }

    #
    # methods required by law
    #

    def output(self):
        """
        Organize output as a (nested) dictionary. Output files will be in a single
        directory, which is determined by `store_parts`.
        """
        return {
            key: self.target(f"{key}.pickle")
            for key in self.output_base_keys
        }

    def requires(self):
        return self.reqs.JER.req_different_branching(self, branch=-1)

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["key"] = self.requires_from_branch()
        return reqs

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

    def dump_output(self, key: str, obj: object):
        if key not in self.output_base_keys:
            raise ValueError(
                f"output key '{key}' not registered in "
                f"`{self.task_family}.output_base_keys`",
            )
        self.output()[key].dump(obj, formatter="pickle")

    #
    # task implementation
    #

    def run(self):
        print(
            f"computing SF for samples {self.branch_data.mc_sample!r} (MC), "
            f"{self.branch_data.data_sample!r} (data)",
        )

        # load JER in data and MC
        h_data = self.load_input("jer", sample=self.branch_data.data_sample)
        h_mc = self.load_input("jer", sample=self.branch_data.mc_sample)

        # collect postprocessor inputs in dict
        hists = {
            "jer_data": h_data,
            "jer_mc": h_mc,
        }

        # run post-processing step for calculating JER
        hists.update(self.postprocessor_inst.run_step(
            task=self,
            step="calc_sf",
            inputs=hists,
        ))

        # store outputs in pickle file for further processing
        for key in self.output_base_keys:
            self.dump_output(key, obj=hists[key])
