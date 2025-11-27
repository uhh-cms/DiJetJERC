# coding: utf-8

"""
Custom tasks to derive JER SF.
"""
from __future__ import annotations

import law

from columnflow.tasks.framework.base import Requirements
from columnflow.util import maybe_import

from dijet.tasks.base import HistogramsBaseTask
from dijet.tasks.alpha import AlphaExtrapolation

hist = maybe_import("hist")
np = maybe_import("numpy")


class JER(
    HistogramsBaseTask,
    law.LocalWorkflow,
):
    """
    Task to calculate JER after alpha extrapolation.

    Processing steps:
    - read in extrapolated widths from `AlphaExtrapolation` task
    - optionally subtract gen-level widths (PLI) from reco (is data use gen-width from MC)
    - calculate JER in using standard method (SM) and forward-extension (FE) methods
    """

    # declare output collection type and keys
    output_collection_cls = law.NestedSiblingFileCollection

    # how to create the branch map
    branching_type = "with_mc"

    # post-processor steps
    postprocessor_steps = ("calc_jer",)

    # upstream requirements
    reqs = Requirements(
        AlphaExtrapolation=AlphaExtrapolation,
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
                output_key: self.target(f"{'__'.join(output_key.split('.'))}.pickle")
                for output_key in self.output_keys
            },
        }

    def requires(self):
        deps = {}

        # require extrapolation results
        deps["reco"] = self.reqs.AlphaExtrapolation.req(self)

        # also require gen-level extrapolation results in MC,
        # in case PLI-subtraction is requested
        # TODO: determine automatically from post-processor?
        mc_samples = [
            b.sample for b in self.branch_map.values()
            if b.is_mc
        ] if self.is_workflow() else [
            self.branch_data.mc_sample,
        ]
        assert len(mc_samples) == 1, "internal error"
        deps["gen"] = self.reqs.AlphaExtrapolation.req_different_branching(
            self,
            samples=mc_samples[0],
            levels="gen",
            branch=0,
        )

        return deps

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

    def load_input(self, input_key: str, level: str, sample: str | None = None):
        sample = sample or self.branch_data.sample
        return self.input()[level][sample][input_key][level].load(formatter="pickle")

    def dump_output(self, output_key: str, obj: object):
        """
        Helper function for writing output to pickle file at the appropriate path.
        """

        # details of path in nested output dict
        path = [
            ("sample", self.branch_data.sample),
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
        sample = self.branch_data.sample
        print(f"computing JER for {sample = }")

        # main response for calculating JER
        response_key = self.postprocessor_inst.calc_jer_main_response

        # load extrapolation results
        input_extp_reco = self.load_input(
            f"{response_key}.extrapolation", level="reco")
        input_extp_gen = self.load_input(
            f"{response_key}.extrapolation", level="gen", sample=self.branch_data.mc_sample)
        input_width_reco = self.load_input(
            f"{response_key}.width", level="reco")
        input_width_gen = self.load_input(
            f"{response_key}.width", level="gen", sample=self.branch_data.mc_sample)

        # load intercept results into hists
        hists = {
            f"{response_key}.width.reco": input_width_reco,
            f"{response_key}.width.gen": input_width_gen,
            f"{response_key}.extrapolation.reco.intercepts": input_extp_reco["intercepts"],
            f"{response_key}.extrapolation.gen.intercepts": input_extp_gen["intercepts"],
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
