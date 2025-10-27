# coding: utf-8

"""
Custom tasks to derive JER SF.
"""
from __future__ import annotations

import law

from columnflow.util import maybe_import
from columnflow.tasks.framework.base import Requirements

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

    # upstream requirements
    reqs = Requirements(
        AlphaExtrapolation=AlphaExtrapolation,
    )

    @property
    def output_base_keys(self):
        return {
            output
            for step in ("calc_jer",)
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
            key: {
                self.branch_data.sample: self.target(f"{key}.pickle"),
            }
            for key in self.output_base_keys
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

    def load_input(self, key: str, level: str, sample: str | None = None):
        sample = sample or self.branch_data.sample
        return self.input()[level][key][sample][level].load(formatter="pickle")

    def dump_output(self, key: str, obj: object):
        if key not in self.output_base_keys:
            raise ValueError(
                f"output key '{key}' not registered in "
                f"`{self.task_family}.output_base_keys`",
            )
        self.output()[key][self.branch_data.sample].dump(obj, formatter="pickle")

    #
    # task implementation
    #

    def run(self):
        sample = self.branch_data.sample
        print(f"computing JER for {sample = }")

        # load extrapolation results
        input_reco = self.load_input("extrapolation", level="reco")
        input_gen = self.load_input("extrapolation", level="gen", sample=self.branch_data.mc_sample)

        # load intercept results into hists
        hists = {
            "extrapolation_reco": input_reco["intercepts"],
            "extrapolation_gen": input_gen["intercepts"],
        }

        # run post-processing step for calculating JER
        hists.update(self.postprocessor_inst.run_step(
            task=self,
            step="calc_jer",
            inputs=hists,
        ))

        # store outputs in pickle file for further processing
        for key in self.output_base_keys:
            self.dump_output(key, obj=hists[key])
