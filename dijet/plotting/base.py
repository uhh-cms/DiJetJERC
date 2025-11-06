# coding: utf-8
"""
Custom base task for plotting workflow steps from JER SF measurement.
"""
from __future__ import annotations

import law

from columnflow.util import DotDict
from columnflow.tasks.framework.parameters import MultiSettingsParameter
from columnflow.tasks.framework.remote import RemoteWorkflow
from dijet.tasks.base import HistogramsBaseTask

from dijet.constants import eta
from dijet.plotting.util import get_bin_slug

logger = law.logger.get_logger(__name__)


# helper function for skipping bins
def bin_skip_fn(val, minmax):
    """
    Return true if the bin described by *val* is to be skipped. The *minmax* parameter is a
    2-tuple indicating the bounds of the range of value permitted to be contained in a bin.

    The supplied *val* is compared to the *minmax* bounds to make the decision. It can be a
    single value or a 2-tuple, in which case the first (second) entry should contain the value
    to be compared to the upper (lower) bound. A bin is skipped if the (second) value is strictly
    less than the lower bound or if the (first) value is strictly larger than the upper bound.

    If *minmax* contains None, the respective bound is not checked.

    The following sketch illustrates bins that would be skipped, assuming that *val* contains
    the lower and upper bounds of each bin:

        val[0]
        |   val[1]
        |   |
    +---+---+---+---+---+---+---+
    | X |   |   |   |   | X | X |
    +---+---+---+---+---+---+---+
        |             |
        minmax[0]     minmax[1]

    """
    # ensure 'val' is a 2-tuple, containing the values to compare to the minimum and the maximum, respectively
    if not isinstance(val, tuple):
        val = (val, val)
    if len(val) != 2:
        raise ValueError(f"internal error: expected 'val' to be a 2-tuple, got: {val}")

    if minmax[0] is not None and val[1] < minmax[0]:
        return True
    elif minmax[1] is not None and val[0] > minmax[1]:
        return True
    else:
        return False


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

    # plot configuration (e.g. axes limits/labels/scales)
    # set in derived tasks
    plot_settings = None

    # variable settings, used to indicate minimum/maximum value of a variable
    # for which to create plots
    bin_selectors = MultiSettingsParameter(
        default=None,
        description="map of variable names with 'min' and 'max' keys to indicate which bins "
        "of those variables to plot; the values given need not be aligned to the bin edges, "
        "the bin containing the value 'min' ('max') will be taken as the first (last) bin to "
        "be plotted",
    )
    # additional non-binning variables that can be specified in --variable-settings (e.g. alpha)
    add_variables_for_settings = {}
    allowed_settings = {"min", "max"}

    #
    # methods required by law
    #

    @classmethod
    def resolve_param_values(cls, params):
        """
        Resolve variable names to corresponding `order.Variable` instances
        and set `variables` param to be passed to dependent histogram task.
        """
        # call super
        params = super().resolve_param_values(params)

        # if not bin selectors provided, set to empty dict and return
        if params["bin_selectors"] is None:
            params["bin_selectors"] = {}
            return params

        # get all variables
        all_variables = set(params["binning_variables"])

        # allowed variables
        if cls.add_variables_for_settings is not None:
            all_variables |= set(cls.add_variables_for_settings)

        # warn about unknown/disallowed variables
        unknown_variables = set(params["bin_selectors"]) - all_variables
        if unknown_variables:
            unknown_variables_str = ",".join(sorted(unknown_variables))
            allowed_variables_str = ",".join(sorted(all_variables))
            logger.warning_once(
                "variables specified in variables_settings are not known or valid "
                f"for use with this task and will be ignored: {unknown_variables_str}; "
                f"allowed variables are: {allowed_variables_str}",
            )

        # drop invalid variables from settings
        params["bin_selectors"] = {
            variable: bin_selectors
            for variable, bin_selectors in params["bin_selectors"].items()
            if variable in all_variables
        }

        # warn if unknown keys are present
        allowed_keys_str = ",".join(sorted(cls.allowed_settings))
        for variable, bin_selectors in params["bin_selectors"].items():
            keys = set(bin_selectors)
            unknown_keys = keys - cls.allowed_settings
            if unknown_keys:
                unknown_keys_str = ",".join(sorted(unknown_keys))
                logger.warning_once(
                    f"settings for variable '{variable}' contain invalid keys, "
                    f"which will be ignored: {unknown_keys_str}; "
                    f"allowed keys are: {allowed_keys_str}",
                )

        return params

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

        # limit variable range for plotting if configured
        # FIXME: avoid hard-coding 'probejet_abseta'?
        eta_minmax = (
            self.bin_selectors.get("probejet_abseta", {}).get("min", None),
            self.bin_selectors.get("probejet_abseta", {}).get("max", None),
        )

        branches = []
        for ib in input_branches:
            branches.extend([
                DotDict.wrap(dict(ib, **{
                    "eta": (eta_lo, eta_hi),
                }))
                for eta_lo, eta_hi in zip(eta[:-1], eta[1:])
                if not bin_skip_fn((eta_lo, eta_hi), eta_minmax)
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
        return {
            "input_task": self.reqs[self.input_task_cls].req_different_branching(self, branch=-1),
        }

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

    #
    # other methods
    #

    def apply_plot_settings(self, ax, context=None):
        if self.plot_settings is None:
            return

        # apply properties to the axes
        for axis in ("x", "y"):
            for prop in ("label", "lim", "scale"):
                key = f"{axis}{prop}"
                value = self.plot_settings.get(key, None)

                # resolve callables
                if callable(value):
                    value = value(self, context or {})

                # call axes method
                if value is not None:
                    getattr(ax, f"set_{key}")(value)
