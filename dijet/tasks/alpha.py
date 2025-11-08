# coding: utf-8

"""
Custom tasks to derive JER SF.
"""
from __future__ import annotations

import law

from columnflow.tasks.framework.base import Requirements
from columnflow.util import maybe_import

from dijet.tasks.base import HistogramsBaseTask
from dijet.tasks.asymmetry import Asymmetry

hist = maybe_import("hist")
np = maybe_import("numpy")
it = maybe_import("itertools")


class AlphaExtrapolation(
    HistogramsBaseTask,
    law.LocalWorkflow,
):
    """
    Task to perform alpha extrapolation.
    Read in and plot asymmetry histograms.
    Extrapolate sigma_A( alpha->0 ).

    Processing steps:
    - read in prepared asymmetry distributions from `Asymmetry` task
    - extract asymmetry widths
    - perform extrapolation of widths to alpha=0 via linear fit including
      correlations
    """

    # declare output collection type and keys
    output_collection_cls = law.NestedSiblingFileCollection

    # how to create the branch map
    branching_type = "separate"

    # post-processor steps
    postprocessor_steps = ("extract_width", "extrapolate_width")

    # upstream requirements
    reqs = Requirements(
        Asymmetry=Asymmetry,
    )

    #
    # methods required by law
    #

    def output(self):
        """Output has same structure as `Asymmetry` task."""
        return self.reqs.Asymmetry.output(self)

    def requires(self):
        """Require `Asymmetry` task."""
        return self.reqs.Asymmetry.req(self)

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

    def load_input(self, input_key: str, level: str):
        sample = self.branch_data.sample
        return self.input()[sample][input_key][level].load(formatter="pickle")

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

    def _run_impl(self, level: str):
        """
        Implementation of width extrapolation from asymmetry distributions.
        """
        # check provided level
        if level not in ("gen", "reco"):
            raise ValueError(f"invalid level '{level}', expected one of: gen,reco")

        # load inputs
        hists = {
            input_key: self.load_input(input_key, level)
            for input_key in self.input_keys
        }

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
            print(f"performing width extraction and alpha extrapolation for {sample = !r}, {level = !r}")
            results = self._run_impl(level=level)

            print("writing outputs")
            for output_key in self.output_keys:
                result = results[output_key]
                self.dump_output(output_key, level, result)
