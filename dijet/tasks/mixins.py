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

from columnflow.types import Any

from columnflow.tasks.framework.base import ConfigTask, RESOLVE_DEFAULT
from columnflow.tasks.framework.mixins import DatasetsProcessesMixin, VariablesMixin
from columnflow.tasks.framework.parameters import DerivableInstParameter

from dijet.postprocessing import PostProcessor
from dijet.util import get_variable_for_level


# TODO: replace with post-processor-based variable resolution
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

    def _make_var_lookup(self, level: str):
        """
        Return a dictionary storing either variables or their gen-level equivalents
        depending on the *level*. Provided for convenient and efficient access to the
        actual variable names
        """
        resolve_var = partial(
            get_variable_for_level,
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
                get_variable_for_level(config_inst, e, level)
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
                names=sample_datasets,
                container=config,
                object_cls=od.Dataset,
                groups_str="dataset_groups",
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


class PostProcessorMixin(ConfigTask):
    """
    Mixin to include a postprocessor into tasks.

    Inheriting from this mixin will allow a task to instantiate and access a
    :py:class:`~dijet.postprocessing.PostProcessor` instance with name *postprocessor*,
    which is an input parameter for this task.
    """

    postprocessor = luigi.Parameter(
        default=RESOLVE_DEFAULT,
        description="the name of the post-processor to be applied; default: value of the "
        "'default_postprocessor' analysis aux",
    )
    postprocessor_inst = DerivableInstParameter(
        default=None,
        visibility=luigi.parameter.ParameterVisibility.PRIVATE,
    )

    allow_empty_postprocessor = False

    exclude_params_index = {"postprocessor_inst"}
    exclude_params_repr = {"postprocessor_inst"}
    exclude_params_sandbox = {"postprocessor_inst"}
    exclude_params_remote_workflow = {"postprocessor_inst"}
    exclude_params_repr_empty = {"postprocessor"}

    @property
    def postprocessor_repr(self) -> str:
        """
        Returns a string representation of the post-processor instance.
        """
        return self.build_repr(str(self.postprocessor_inst))

    @classmethod
    def req_params(cls, inst: law.Task, **kwargs) -> dict[str, Any]:
        # prefer --postprocessor set on task-level via cli
        kwargs["_prefer_cli"] = law.util.make_set(kwargs.get("_prefer_cli", [])) | {"postprocessor"}
        return super().req_params(inst, **kwargs)

    @classmethod
    def get_postprocessor_inst(
        cls,
        postprocessor: str,
        analysis_inst: od.Analysis,
        requested_configs: list[str] | None = None,
        **kwargs,
    ) -> PostProcessor:
        """
        Get requested *postprocessor* instance.

        This method retrieves the requested *postprocessor* instance. If *requested_configs* are provided,
        they are passed to the setup function of the post-processor.

        :param postprocessor: Name of :py:class:`~dijet.postprocessing.PostProcessor` to load.
        :param analysis_inst: Forward this analysis inst to the init function of new PostProcessor sub class.
        :param requested_configs: Configs passed to the processor.
        :param kwargs: Additional keyword arguments to forward to the :py:class:`~dijet.postprocessing.PostProcessor`
                       instance.
        :return: :py:class:`~dijet.postprocessing.PostProcessor` instance.
        """
        postprocessor_inst: PostProcessor = PostProcessor.get_cls(postprocessor)(analysis_inst, **kwargs)
        if requested_configs:
            postprocessor_inst._setup(requested_configs)

        return postprocessor_inst

    @classmethod
    def get_config_lookup_keys(
        cls,
        inst_or_params: PostProcessorMixin | dict[str, Any],
    ) -> law.util.InsertiableDict:
        keys = super().get_config_lookup_keys(inst_or_params)

        # add the post-processor name
        postprocessor = (
            inst_or_params.get("postprocessor")
            if isinstance(inst_or_params, dict)
            else getattr(inst_or_params, "postprocessor", None)
        )
        if postprocessor not in (law.NO_STR, None, ""):
            prefix = "pp"
            keys[prefix] = f"{prefix}_{postprocessor}"

        return keys

    @classmethod
    def resolve_param_values_pre_init(cls, params: dict[str, Any]) -> dict[str, Any]:
        params = super().resolve_param_values_pre_init(params)

        # add the default post-processor when empty
        if (container := cls._get_config_container(params)):
            params["postprocessor"] = cls.resolve_config_default(
                param=params.get("postprocessor"),
                task_params=params,
                container=container,
                default_str="default_postprocessor",
                multi_strategy="same",
            )

        # when both config_inst and postprocessor are set, initialize the postprocessor_inst
        if all(params.get(x) not in {None, law.NO_STR} for x in ("config_inst", "postprocessor")):
            if not params.get("postprocessor_inst"):
                params["postprocessor_inst"] = cls.get_postprocessor_inst(
                    params["postprocessor"],
                    params["analysis_inst"],
                    requested_configs=[params["config_inst"]],
                )
        elif not cls.allow_empty_postprocessor:
            raise Exception(f"no postprocessor configured for {cls.task_family}")

        return params

    def store_parts(self) -> law.util.InsertableDict:
        parts = super().store_parts()

        if self.postprocessor_inst:
            parts.insert_before("version", "postprocessor", f"pp__{self.postprocessor_repr}")

        return parts
