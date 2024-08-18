# coding: utf-8

"""
Custom mixins for adding parameters/methods specific to
dijet analysis.
"""
from __future__ import annotations

from functools import partial

import luigi
import law
import order as od

from columnflow.tasks.framework.base import ConfigTask
from columnflow.tasks.framework.mixins import DatasetsProcessesMixin, VariablesMixin


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
        "if not given, uses default asymmetry variable specified in config; empty default",
        default=None,
    )
    alpha_variable = luigi.Parameter(
        description="variable used to quantify the third jet activity (e.g. 'dijets_alpha'); "
        "if not given, uses default alpha variable specified in config; empty default",
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
    variable_param_names = (
        "asymmetry_variable", "alpha_variable", "binning_variables",
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # store `od.Variable`` instances of variables
        self.alpha_variable_inst = self.config_inst.get_variable(self.alpha_variable)
        self.asymmetry_variable_inst = self.config_inst.get_variable(self.asymmetry_variable)
        self.binning_variable_insts = {
            v: self.config_inst.get_variable(v)
            for v in self.binning_variables
        }

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

    def _make_var_lookup(self, level: str):
        """
        Return a dictionary storing either variables or their gen-level equivalents
        depending on the *level*. Provided for convenient and efficient access to the
        actual variable names
        """
        resolve_var = partial(
            self._get_variable_for_level,
            config=self.config_inst,
            level=level,
        )
        return {
            "alpha": resolve_var(name=self.alpha_variable),
            "asymmetry": resolve_var(name=self.asymmetry_variable),
            "binning": {
                bv: resolve_var(name=bv)
                for bv in self.binning_variables
            },
        }

    def get_level_index(self, level: str):
        if level not in self.levels:
            raise ValueError(
                f"unknown level '{level}', valid: {','.join(self.levels)}",
            )
        return self.levels.index(level)

    def iter_levels_variables(self, levels: list[str] | None = None):
        """
        Generator yielding tuples of the form (level, variable), with *level*
        being either 'reco' for reconstruction-level or 'gen' for gen-level
        variables. An *levels* argument can be provided to restrict the levels
        (e.g. gen-level in MC).
        """
        assert len(self.levels) == len(self.variables)
        for level, variable in zip(self.levels, self.variables):
            if levels and level not in levels:
                continue
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


class DiJetSamplesMixin(ConfigTask):
    """
    Custom mixin for dijet samples/datasets. A 'sample' in the context of the dijet
    analysis is a collection of standard 'columnflow' datasets to be treated as
    a single unit. For instance, a "qcd_ht" sample could comprise all/several HT bins
    of the QCD multijet MC datasets, which in columflow would correspond to all
    datasets with names mathing 'qcd_ht*'. In that sense, 'samples' function similarly
    to 'dataset_groups' in standard columnflow.

    Samples (along with corresponding datasets and other metadata) must be specified
    in the configuration under the aux key 'samples'. An example is given below:

    .. code-block:: python

        cfg.x.samples = {
            "data": {
                "datasets": "data_*",
                "color": "k",
            },
            "qcdht": {
                "datasets": "qcd_*",
                "color": "indianred",
            },
        }
    """

    # make `datasets` parameter private -> value computed based on other variable parameters
    datasets = DatasetsProcessesMixin.datasets.copy()
    datasets.visibility = luigi.parameter.ParameterVisibility.PRIVATE

    samples = law.CSVParameter(
        default=(),
        description="comma-separated sample names; for each sample, a mapping to a corresponding "
        "list of datasets must be defined in the auxiliary data of the config under the key "
        "'samples'; mapped datasets can also be patterns or keys of a mapping defined in the "
        "'dataset_groups' auxiliary data of the config; when empty, uses all samples registered in the "
        "config; empty default",
        brace_expand=True,
        parse_empty=True,
    )

    allow_empty_datasets = False

    @classmethod
    def resolve_param_values(cls, params):
        # note: implementation similar to `DatasetsProcessesMixin`,
        # but without setting `processes` and resolving the `datasets`
        # based on the `samples` dictionary in the config
        params = super().resolve_param_values(params)

        if "config_inst" not in params:
            return params
        config_inst = params["config_inst"]

        # resolve datasets for samples
        if "samples" in params:
            # resolve datasets
            datasets = cls.get_datasets(config_inst, params["samples"], allow_empty=False)

            # complain when no datasets were found
            if not datasets and not cls.allow_empty_datasets:
                raise ValueError(f"no datasets found matching samples {params['samples']}")

            params["datasets"] = tuple(datasets)
            params["dataset_insts"] = [config_inst.get_dataset(d) for d in params["datasets"]]

        return params

    @classmethod
    def get_datasets(cls, config: od.Config, samples: list[str], allow_empty: bool = True):
        """
        Resolve `samples` to actual `datasets`.
        """
        # get list of datasets for sample
        all_samples = {
            sample: sample_cfg.get("datasets", [])
            for sample, sample_cfg in config.x("samples", {}).items()
        }

        all_datasets = []
        empty_samples = set()
        for sample in samples:
            sample_datasets = all_samples.get(sample, [])

            # resolve patterns/dataset groups
            sample_datasets = cls.find_config_objects(
                sample_datasets,
                config,
                od.Dataset,
                config.x("dataset_groups", {}),
            )

            # keep track of empty samples
            if not sample_datasets:
                empty_samples.add(sample)

            # add to list of returned datasets
            all_datasets.extend(sample_datasets)

        # raise exception when sample is not registered or
        # has no datasets
        if empty_samples and not allow_empty:
            empty_samples_str = ",".join(sorted(empty_samples))
            raise ValueError(f"no datasets found matching samples: {empty_samples_str}")

        return all_datasets

    @staticmethod
    def get_samples_repr(samples: list[str]):
        if len(samples) <= 2:
            return "_".join(samples)

        return f"{len(samples)}_{law.util.create_hash(sorted(samples))}"

    @property
    def samples_repr(self):
        return self.get_samples_repr(self.samples)

    def get_sample_index(self, sample: str):
        if sample not in self.samples:
            raise ValueError(
                f"unknown sample '{sample}', valid: {','.join(self.samples)}",
            )
        return self.samples.index(sample)
