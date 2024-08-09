# coding: utf-8

"""
Custom base tasks.
"""

import luigi
import law
import order as od

from functools import partial

from columnflow.tasks.framework.base import BaseTask, ShiftTask
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorMixin, ProducersMixin,
    VariablesMixin, DatasetsProcessesMixin, CategoriesMixin,
)
from columnflow.config_util import get_datasets_from_process
from columnflow.util import dev_sandbox, DotDict


class DiJetTask(BaseTask):
    task_namespace = "dijet"


class DiJetVariablesMixin(
    VariablesMixin,
):
    """
    Custom `VariablesMixin` to specify dijet-related variables.

    Computes `variables` parameter for multidimensional histograms
    to be passed on to `MergeHistograms`.
    """

    # make `variables` parameter private -> value computed based on other variable parameters
    variables = VariablesMixin.variables.copy()
    variables.visibility = luigi.parameter.ParameterVisibility.PRIVATE

    asymmetry_variable = luigi.Parameter(
        description="variable used to quantify the dijet response (e.g. 'dijets_asymmetry'); "
        "if not given, uses default response variable specified in config; empty default",
        default=None,
    )
    alpha_variable = luigi.Parameter(
        description="variable used to quantify the third jet activity (e.g. 'dijets_alpha'); "
        "if not given, uses default response variable specified in config; empty default",
        default=None,
    )
    binning_variables = law.CSVParameter(
        default=(),
        description="variables to use for constructing the bins in which the dijet response "
        "should be measured (e.g. 'probejet_eta,dijets_pt_avg'); if not given, uses default "
        "binning variables specified in config; empty default",
        brace_expand=True,
        parse_empty=True,
    )
    levels = law.CSVParameter(
        default=("reco", "gen"),
        description="comma-separated list of 'reco', 'gen', or both, indicating whether to "
        "use the regular (reconstruction-level) variables or the equivalent variable on "
        "generator-level; if not given, only 'reco'-level variables are used",
        choices={"reco", "gen"},
        brace_expand=True,
        parse_empty=True,
    )
    variable_param_names = (
        "asymmetry_variable", "alpha_variable", "binning_variables",
    )

    @staticmethod
    def _get_variable_for_level(config: od.Config, name: str, level: str):
        """
        Get name of variable corresponding to main variable *name*
        of level *level*, where *level* can be either 'reco' or 'gen'.
        """
        if level == "reco":
            # reco level is default -> return directly
            return name
        elif level == "gen":
            # look up registered gen-level name in aux data
            var_inst = config.get_variable(name)
            return var_inst.x("gen_variable", name)
        else:
            raise ValueError(f"invalid level '{level}', expected one of: gen,reco")

    def iter_levels_variables(self):
        """
        Generator yielding tuples of the form (level, variable), with *level*
        being either 'reco' for reconstruction-level or 'gen' for gen-level
        variables.
        """
        assert len(self.levels) == len(self.variables)
        for level, variable in zip(self.levels, self.variables):
            yield level, variable

    @classmethod
    def resolve_param_values(cls, params):
        """
        Resolve variable names to corresponding `order.Variable` instances
        and set `variables` param to be passed to dependent histogram task.
        """
        # call super
        params = super().resolve_param_values(params)

        # return early if config and dataset instances not loaded yet
        # or if required parameters are not present
        if any(
            p not in params
            for p in (
                "config_inst",
            ) + cls.variable_param_names
        ):
            return params

        # dijet variables already resolved, do nothing
        if params.get("_dijet_vars_resolved", False):
            return params

        # get config and dataset instances
        config_inst = params["config_inst"]

        # resolve variable defaults from config
        for var_param_name in cls.variable_param_names:
            params[var_param_name] = params[var_param_name] or (
                config_inst.x(f"default_{var_param_name}", None)
            )

        # raise error if unresolved defaults
        missing_param_names = {
            var_param_name
            for var_param_name in cls.variable_param_names
            if params[var_param_name] is None
        }
        if missing_param_names:
            raise RuntimeError(
                "could not find default values in config for DiJetVariablesMixin; "
                "please define the following keys in your config:\n" + "\n".join(
                    f"default_{var_name}" for var_name in sorted(missing_param_names)
                ),
            )

        # construct multi-dimensional variables
        multivar_elems = [
            params["alpha_variable"],
        ] + list(
            params["binning_variables"],
        ) + [
            params["asymmetry_variable"],
        ]
        multivars = []
        for level in params["levels"]:
            # required variables
            multivar_elems_for_level = [
                cls._get_variable_for_level(config_inst, e, level)
                for e in multivar_elems
            ]
            multivars.append(
                cls.join_multi_variable(multivar_elems_for_level),
            )

        # convert to tuple and register as parameter
        params["variables"] = law.util.make_tuple(multivars)

        # set flag to prevent function from running again
        params["_dijet_vars_resolved"] = True

        return params


class HistogramsBaseTask(
    DiJetTask,
    DatasetsProcessesMixin,
    CategoriesMixin,
    DiJetVariablesMixin,
    ProducersMixin,
    SelectorMixin,
    CalibratorsMixin,
    ShiftTask,
    law.LocalWorkflow,
):
    """
    Base task to load histogram and reduce them to used information.
    An example implementation of how to handle the inputs in a run method can be
    found in columnflow/tasks/histograms.py
    """
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # Add nested sibling directories to output path
    output_collection_cls = law.NestedSiblingFileCollection

    # Category ID for methods
    LOOKUP_CATEGORY_ID = {"sm": 1, "fe": 2}

    def get_datasets(self) -> tuple[list[str], bool]:
        """
        Select datasets belonging to the `process` of the current branch task.
        Returns a list of the dataset names and a flag indicating if they are
        data or MC.
        """
        # all datasets in config that are mapped to a process
        dataset_insts_from_process = get_datasets_from_process(
            self.config_inst,
            self.branch_data.process,
            only_first=False,
        )

        # check that at least one config dataset matched
        if not dataset_insts_from_process:
            raise RuntimeError(
                "no single dataset found in config matching "
                f"process `{self.branch_data.process}`",
            )

        # filter to contain only user-supplied datasets
        datasets_from_process = [d.name for d in dataset_insts_from_process]
        datasets_filtered = set(self.datasets).intersection(datasets_from_process)

        # check that at least one user-supplied dataset matched
        if not datasets_filtered:
            raise RuntimeError(
                "no single user-supplied dataset matched "
                f"process `{self.branch_data.process}`",
            )

        # set MC flag if any of the filtered datasets
        datasets_filtered_is_mc = set(
            self.config_inst.get_dataset(d).is_mc
            for d in datasets_filtered
        )
        if len(datasets_filtered_is_mc) > 1:
            raise RuntimeError(
                "filtered datasets have mismatched `is_mc` flags",
            )

        # return filtered datasets and is_mc flag
        return (
            datasets_filtered,
            list(datasets_filtered_is_mc)[0],
        )

    def create_branch_map(self):
        """
        Workflow has one branch for each process supplied via `processes`.
        """
        return [
            DotDict({"process": process})
            for process in sorted(self.processes)
        ]

    def store_parts(self):
        parts = super().store_parts()
        return parts

    def extract_sample(self):
        datasets, isMC = self.get_datasets()
        # Define output name
        if isMC:
            sample = "QCDHT"
        else:
            runs = []
            for dataset in datasets:
                runs.append(dataset.replace("data_jetht_", "").upper())
            sample = "Run" + ("".join(sorted(runs)))
        return sample

    def reduce_histogram(self, histogram, processes, shift, level):
        """
        Reduce away the `shift` and `process` axes of a multidimensional
        histogram by selecting a single shift and summing over the processes.
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

        # transform into list if necessary
        processes = law.util.make_list(processes)

        # get all sub processes
        process_insts = list(map(self.config_inst.get_process, processes))
        sub_process_insts = set(flatten_nested_list([
            [sub for sub, _, _ in proc.walk_processes(include_self=True)]
            for proc in process_insts
        ]))

        # get shift instance
        shift_inst = self.config_inst.get_shift(shift)
        if shift_inst.id not in histogram.axes["shift"]:
            raise ValueError(f"histogram does not contain shift `{shift}`")

        # work on a copy
        h = histogram.copy()

        # axis selections
        h = h[{
            "process": [
                hist.loc(p.id)
                for p in sub_process_insts
                if p.id in h.axes["process"]
            ],
            "shift": hist.loc(shift_inst.id),
        }]

        # TODO: Sum over shift ? Be careful to only take one shift -> Error if more ?
        # axis reductions
        h = h[{
            "process": sum,
            # TODO: read rebinning factors from config
            # @dsavoiu: might be better to use config binning for now
            # vars_["alpha"]: hist.rebin(5),
            vars_["asymmetry"]: hist.rebin(2),
        }]

        return h
