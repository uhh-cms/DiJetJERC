# coding: utf-8
"""
Custom base task for plotting workflow steps from JER SF measurement.
"""
from __future__ import annotations

import law

from columnflow.util import DotDict, maybe_import
from columnflow.types import TYPE_CHECKING
from columnflow.tasks.framework.parameters import MultiSettingsParameter
from columnflow.tasks.framework.remote import RemoteWorkflow
from dijet.tasks.base import HistogramsBaseTask

from dijet.util import product_dict
from dijet.plotting.util import get_bin_label, get_bin_slug

logger = law.logger.get_logger(__name__)

if TYPE_CHECKING:
    hist = maybe_import("hist")


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
        description="map of variable keys with 'min' and 'max' keys to indicate which bins "
        "of those variables to plot; the values given need not be aligned to the bin edges, "
        "the bin containing the value 'min' ('max') will be taken as the first (last) bin to "
        "be plotted",
    )
    # keys that users are allowed to pass in `--bin-selectors`
    bin_selectors_allowed_keys = {"min", "max"}

    # which variable keys to use for constructing branch map
    # (i.e. one branch will be created for each element of the
    # cartesian product of bins in these variables)
    # TODO: avoid hardcoding variable keys?
    # set in derived tasks
    branch_map_binning_variable_keys = None

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

        # check postprocessor exists and defines a variable_map
        if (postprocessor_inst := params.get("postprocessor_inst", None)) is None:
            raise RuntimeError(
                "internal error: task does not have an initialized postprocessor instance, aborting",
            )
        if not hasattr(postprocessor_inst, "variable_map"):
            raise RuntimeError(
                "postprocessor instance does not define a `variable_map`, aborting",
            )

        # get all variable keys
        all_variable_keys = set(params["postprocessor_inst"].variable_map["reco"])

        # warn about unknown/disallowed variables
        unknown_keys = set(params["bin_selectors"]) - all_variable_keys
        if unknown_keys:
            unknown_keys_str = ",".join(sorted(unknown_keys))
            allowed_keys_str = ",".join(sorted(all_variable_keys))
            logger.warning_once(
                "variables specified in variables_settings are not known or valid "
                f"for use with this postprocessor and will be ignored: {unknown_keys_str}; "
                f"allowed variable specifiers are: {allowed_keys_str}",
            )

        # drop invalid variables from settings
        params["bin_selectors"] = {
            variable: bin_selectors
            for variable, bin_selectors in params["bin_selectors"].items()
            if variable in all_variable_keys
        }

        # warn if unknown keys are present
        allowed_keys_str = ",".join(sorted(cls.bin_selectors_allowed_keys))
        for variable, bin_selectors in params["bin_selectors"].items():
            keys = set(bin_selectors)
            unknown_keys = keys - cls.bin_selectors_allowed_keys
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

    def _get_filtered_variable_bins(
        self,
        variable_map: dict[str],
        variable_keys: list[str],
        bin_selectors: dict[dict] | None = None,
        from_hist: hist.Hist | None = None,
        remove_skipped: bool = True,
    ) -> dict[list[dict]]:
        """
        Given a variable map and a sequence of keys, return a dict mapping
        those keys to a list of bin dicts, each containing information
        like upper and lower bin egdes, bin center or a human-readable
        string representing the bin (a.k.a a 'slug').

        By default, this function uses `od.Variable` information contained in
        via the *config_inst*. If `from_hist` is provided, the binning is
        retrieved from the axes of the given `hist.Hist` object. Note that
        the histogram must contain the axes contained in the `variable_map`.

        If `remove_skipped` is True, skipped bins will not appear in the output.
        Otherwise, they will be included and marked by a `skip=True` entry
        in the bin dict.
        """
        # use default bin selectors if not specified
        bin_selectors = bin_selectors or self.bin_selectors

        # retrieve which bins to plot
        bv_bins = {}
        for bv_key in variable_keys:
            # retrieve actual variable name
            bv_name = variable_map[bv_key]

            # retrieve variable instance from config
            bv_inst = self.config_inst.get_variable(bv_name)

            # get min and max binning value for filtering
            bv_minmax = (
                bin_selectors.get(bv_key, {}).get("min", None),
                bin_selectors.get(bv_key, {}).get("max", None),
            )

            # get bin edges from config or input histogram
            if from_hist is None:
                bv_edges = bv_inst.bin_edges
            else:
                bv_edges = from_hist.axes[bv_name].edges

            # get filtered list of bins for the branch map
            bv_bins[bv_key] = [
                {
                    # bin edges and center
                    "lo": bv_lo,
                    "center": 0.5 * (bv_lo + bv_hi),
                    "hi": bv_hi,
                    # representation to use in file names
                    "slug": get_bin_slug(bv_inst, (bv_lo, bv_hi)),
                    "label": get_bin_label(bv_inst, (bv_lo, bv_hi)),
                    # key and name of variable being binned
                    "var_key": bv_key,
                    "var_name": bv_name,
                    # selector to use for slicing histograms
                    "loc": 0.5 * (bv_lo + bv_hi),
                    "skip": bin_skip,
                }
                for bv_lo, bv_hi in zip(bv_edges[:-1], bv_edges[1:])
                if not ((bin_skip := bin_skip_fn((bv_lo, bv_hi), bv_minmax)) and remove_skipped)
            ]

        return bv_bins

    def create_branch_map(self):
        """
        Workflow extends branch map of input task, creating one branch
        per entry in the input task branch map per each |eta| bin (abseta).
        """
        input_branches = super().create_branch_map()

        # limit variable range for plotting if configured
        # FIXME: avoid hard-coding 'reco'?
        variable_map = self.postprocessor_inst.variable_map["reco"]

        if self.branch_map_binning_variable_keys is None:
            raise ValueError(
                "cannot construct branch map: property "
                "'branch_map_binning_variable_keys' not set for task "
                f"{self.__class__.__name__}!",
            )

        # check keys in variable map
        missing_keys = {
            bv_key
            for bv_key in self.branch_map_binning_variable_keys
            if bv_key not in variable_map
        }
        if missing_keys:
            missing_keys_str = ",".join(sorted(missing_keys))
            raise ValueError(
                f"postprocessor '{self.postprocessor}' variable map does not define "
                "variables the following keys, which are needed for creating the "
                f"branch map for plotting: {missing_keys_str}",
            )

        # retrieve which bins to place on branches
        bv_bins = self._get_filtered_variable_bins(
            variable_map,
            self.branch_map_binning_variable_keys,
            from_hist=None,  # uses config for now -> switch to actual inputs?
            remove_skipped=True,
        )

        # extend super branching map by cartesian product
        # of binning
        branches = []
        for ib in input_branches:
            branches.extend([
                DotDict.wrap(dict(ib, **bv_bin_dict))
                for bv_bin_dict in product_dict(bv_bins)
            ])

        return branches

    def output(self) -> dict[law.FileSystemTarget]:
        """
        Organize output as a (nested) dictionary. Output files will be in a single
        directory, which is determined by `store_parts`.
        """
        # join bin slugs to get output directory path
        bin_slug = "/".join(
            self.branch_data[bv_key]["slug"]
            for bv_key in self.branch_map_binning_variable_keys
        )
        return {
            "dummy": self.target(f"{bin_slug}/DUMMY"),
            "plots": self.target(f"{bin_slug}", dir=True),
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

    @property
    def input_keys(self):
        """
        Return all keys to be plotted
        """
        if self.is_branch():
            return self.requires()["input_task"].output_keys
        else:
            return self.requires()["key"]["input_task"].output_keys

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
