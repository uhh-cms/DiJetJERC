# coding: utf-8

"""
Custom tasks to derive JER SF.
"""

import law
import order as od

from columnflow.util import maybe_import
from columnflow.tasks.framework.base import Requirements

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

    # upstream requirements
    reqs = Requirements(
        Asymmetry=Asymmetry,
    )

    @property
    def output_base_keys(self):
        return {
            output
            for step in ("extract_width", "extrapolate_width")
            for output in self.postprocessor_inst.steps[step].outputs
        }

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

    def load_input(self, key: str, level: str):
        return self.input()[key][self.branch_data.sample][level].load(formatter="pickle")

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

    def _run_impl(self, datasets: list[od.Dataset], level: str, variable: str):
        """
        Implementation of width extrapolation from asymmetry distributions.
        """
        # check provided level
        if level not in ("gen", "reco"):
            raise ValueError(f"invalid level '{level}', expected one of: gen,reco")

        # load asymmetry histograms and number of events
        h_asyms = self.load_input("asym_cut", level=level)
        h_nevts = self.load_input("nevt", level=level)

        hists = {
            "asym_cut": h_asyms,
            "nevt": h_nevts,
        }

        # run post-processing step for extracting distributionwidths
        hists.update(self.postprocessor_inst.run_step(
            task=self,
            step="extract_width",
            inputs=hists,
            level=level,
        ))

        # run post-processing step for extrapolating distribution widths
        hists.update(self.postprocessor_inst.run_step(
            task=self,
            step="extrapolate_width",
            inputs=hists,
            level=level,
        ))

        # store outputs in pickle file for further processing
        for key in self.output_base_keys:
            self.dump_output(key, level=level, obj=hists[key])

    def run(self):
        # process histograms for all applicable levels
        sample = self.branch_data.sample
        for level, variable in self.iter_levels_variables():
            print(f"performing alpha extrapolation for {sample = }, {level = }, {variable = }")
            self._run_impl(self.branch_data.datasets, level=level, variable=variable)
