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
