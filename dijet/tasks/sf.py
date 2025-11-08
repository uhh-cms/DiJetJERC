# coding: utf-8

"""
Custom tasks to derive JER SF.
"""
from __future__ import annotations

import law

from columnflow.tasks.framework.base import Requirements
from columnflow.util import maybe_import

from dijet.tasks.base import HistogramsBaseTask
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

    # post-processor steps
    postprocessor_steps = ("calc_sf",)

    # upstream requirements
    reqs = Requirements(
        JER=JER,
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
            # key indicating result produced by processing step
            output_key: self.target(f"{'__'.join(output_key.split('.'))}.pickle")
            for output_key in self.output_keys
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

    def load_input(self, input_key: str, sample: str):
        coll_keys = [
            coll_key
            for coll_key, coll in self.input()["collection"].targets.items()
            if sample in coll
        ]
        if len(coll_keys) != 1:
            raise RuntimeError(
                f"found {len(coll_keys)} input collections corresponding to "
                f"sample '{sample}', expected 1",
            )
        return self.input()["collection"][coll_keys[0]][sample][input_key].load(formatter="pickle")

    def dump_output(self, output_key: str, obj: object):
        """
        Helper function for writing output to pickle file at the appropriate path.
        """

        # details of path in nested output dict
        path = [
            ("output_key", output_key),
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

    def run(self):
        print(
            f"computing SF for samples {self.branch_data.mc_sample!r} (MC), "
            f"{self.branch_data.data_sample!r} (data)",
        )

        # main response for calculating JER
        response_key = self.postprocessor_inst.calc_jer_main_response

        # load JER in data and MC
        h_data = self.load_input(f"{response_key}.jer", sample=self.branch_data.data_sample)
        h_mc = self.load_input(f"{response_key}.jer", sample=self.branch_data.mc_sample)

        # collect postprocessor inputs in dict
        hists = {
            f"{response_key}.jer.data": h_data,
            f"{response_key}.jer.mc": h_mc,
        }

        # run post-processing steps
        for step in self.postprocessor_steps:
            hists.update(self.postprocessor_inst.run_step(
                task=self,
                step=step,
                inputs=hists,
            ))

            print("writing outputs")
            for output_key in self.output_keys:
                result = hists[output_key]
                self.dump_output(output_key, result)
