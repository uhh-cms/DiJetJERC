# coding: utf-8
"""
Custom base task for plotting workflow steps from JER SF measurement.
"""
from __future__ import annotations

import law

from columnflow.util import DotDict
from columnflow.tasks.framework.remote import RemoteWorkflow

from dijet.tasks.base import HistogramsBaseTask

from dijet.constants import eta
from dijet.plotting.util import get_bin_slug


class PlottingBaseTask(
    HistogramsBaseTask,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    """
    Base task to plot histogram from each step of the JER SF workflow.
    An example implementation of how to handle the inputs in a run method can be
    found in columnflow/tasks/histograms.py
    """

    # upstream workflow
    input_task_cls = None  # set this in derived tasks

    # parameters
    file_types = law.CSVParameter(
        default=("pdf",),
        significant=True,
        description="comma-separated list of file extensions to produce; default: pdf",
    )

    #
    # methods required by law
    #

    @classmethod
    @property
    def reqs(cls):
        reqs = super().reqs

        if cls.input_task_cls is not None:
            reqs[cls.input_task_cls] = cls.input_task_cls

        return reqs

    def create_branch_map(self):
        """
        Workflow extends branch map of input task, creating one branch
        per entry in the input task branch map per each eta bin (eta).
        """
        # TODO: way to specify which variables to handle via branch
        # map and which to loop over in `run` method
        # TODO: don't hardcode eta bins, use dynamic workflow condition
        # to read in bins from task inputs
        input_branches = super().create_branch_map()

        branches = []
        for ib in input_branches:
            branches.extend([
                DotDict.wrap(dict(ib, **{
                    "eta": (eta_lo, eta_hi),
                }))
                for eta_lo, eta_hi in zip(eta[:-1], eta[1:])
            ])

        return branches

    def output(self) -> dict[law.FileSystemTarget]:
        """
        Organize output as a (nested) dictionary. Output files will be in a single
        directory, which is determined by `store_parts`.
        """
        eta_bin_slug = get_bin_slug(self.binning_variable_insts["probejet_abseta"], self.branch_data.eta)
        return {
            "dummy": self.target(f"{eta_bin_slug}/DUMMY"),
            "plots": self.target(f"{eta_bin_slug}", dir=True),
        }

    def requires(self):
        return self.reqs[self.input_task_cls].req_different_branching(self, branch=-1)

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["key"] = self.requires_from_branch()
        return reqs

    #
    # helper methods for handling task inputs/outputs
    #

    def save_plot(self, basename: str, fig: object, extensions: tuple[str] | list[str] | None = None):
        for ext in self.file_types:
            target = self.output()["plots"].child(f"{basename}.{ext}", type="f")
            target.dump(fig, formatter="mpl")
            print(f"saved plot: {target.path}")
