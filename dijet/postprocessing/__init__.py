# coding: utf-8
"""
Tools for post-processing of histograms.
"""
from __future__ import annotations

import law
import order as od

from columnflow.types import Callable
from columnflow.util import maybe_import, Derivable, DotDict
from columnflow.types import TYPE_CHECKING, Any, Type, T

if TYPE_CHECKING:
    ak = maybe_import("awkward")
    hist = maybe_import("hist")


class PostProcessor(Derivable):
    """
    Object representing a JER postprocessor. Works analogously to :py:class:`TaskArrayFunction`,
    except that it acts on and produces histograms instead of event arrays.

    Inheriting classes also need to overwrite the following abstract methods:
        - `variables`: return a sequence of multi-dimensional variables used to fill input histograms

    There are several optional hooks:
        - :py:meth:`step`: register a function that represents a post-processing step; called during
                           execution post-processing tasks

    There are several optional hooks:
        - :py:meth:`setup`: a custom setup after config objects were assigned
        - :py:meth:`requires`: additional tasks required for post-processor
    """

    # default setting mark whether this model accepts only a single config
    single_config: bool = True

    # default name for storing e.g. input data
    # falls back to cls_name if None
    store_name: str | None = None

    # names of attributes that are automatically extracted from init kwargs and
    # fall back to classmembers in case they are missing
    init_attributes: list[str] = ["single_config", "store_name"]

    @classmethod
    def postprocessor(
        cls: T,
        name: str,
        bases: tuple[type] = (),
        **kwargs,
    ) -> Type[T] | Callable:
        """
        Decorator for creating a new :py:class:`PostProcessor` subclass with additional, optional
        *bases*. A *variables* dictionary that maps keys to variable names must be provided.
        Additional *kwargs* are added as class members of the new subclass.

        :param func: The function to be decorated and attached as ``init_func``.
        :param bases: Optional tuple of base classes for the new subclass.
        :returns: The new subclass or a decorator function.
        """
        def decorator() -> Type[T]:
            # create the class dict
            cls_dict = {
                **kwargs,
                "steps": DotDict.wrap({}),  # storage for post-processing steps
            }

            # create the subclass
            subclass = cls.derive(name, bases=bases, cls_dict=cls_dict)

            return subclass

        return decorator()

    def __init__(
        self: PostProcessor,
        analysis_inst: od.Analysis,
        **kwargs,
    ) -> None:
        super().__init__()

        # store attributes
        self.analysis_inst = analysis_inst

        # set instance members based on registered init attributes
        for attr in self.init_attributes:
            # get the class-level attribute
            value = getattr(self, attr)
            # get the value from kwargs
            _value = kwargs.get(attr, law.no_value)
            if _value != law.no_value:
                value = _value
            # set the instance-level attribute
            setattr(self, attr, value)

        # list of config instances
        self.config_insts = []
        if "configs" in kwargs:
            self._setup(kwargs["configs"])

    @classmethod
    def step(
        cls,
        name: str,
        func: Callable | None = None,
        inputs: set[str] | None = None,
        outputs: set[str] | None = None,
        **kwargs,
    ) -> None:
        """
        Decorator to wrap a function *func* that should be registered as a post-processing step. The
        decorated function is added to the class-level :py:attr:`steps` dictionary using its name
        as a key.
        """
        def decorator(func: Callable) -> Callable:
            if name in cls.steps:
                raise ValueError(f"post-processing step '{name}' already registered")

            expanded_inputs = {
                input_
                for input_spec in inputs
                for input_ in law.util.brace_expand(input_spec)
            }
            expanded_outputs = {
                output
                for output_spec in outputs
                for output in law.util.brace_expand(output_spec)
            }

            cls.steps[name] = DotDict.wrap({
                "func": func,
                "inputs": expanded_inputs,
                "outputs": expanded_outputs,
                **kwargs,
            })

            return func

        return decorator(func) if func else decorator

    def __str__(self):
        """
        Returns a string representation of this post-processor. Defaults to the class name.
        """
        return self.cls_name

    @property
    def config_inst(self: PostProcessor) -> od.Config:
        if self.single_config and len(self.config_insts) != 1:
            raise Exception(
                f"the config_inst property requires PostProcessor '{self.cls_name}' to have the "
                "single_config enabled to to contain exactly one config instance, but found "
                f"{len(self.config_insts)}",
            )

        return self.config_insts[0]

    def _assert_configs(self: PostProcessor, msg: str) -> None:
        """
        Raises an exception showing *msg* in case this model's :py:attr:`config_insts` is empty.
        """
        if not self.config_insts:
            raise Exception(f"PostProcessor '{self.cls_name}' has no config instances, {msg}")

    def _set_configs(self: PostProcessor, configs: list[str | od.Config]) -> None:
        # complain when only a single config is accepted
        if self.single_config and len(configs) > 1:
            raise Exception(
                f"PostProcessor '{self.cls_name}' only accepts a single config but received "
                f"{len(configs)}: {','.join(map(str, configs))}",
            )

        # remove existing config instances
        del self.config_insts[:]

        # add them one by one
        for config in configs:
            config_inst = (
                config
                if isinstance(config, od.Config)
                else self.analysis_inst.get_config(config)
            )
            self.config_insts.append(config_inst)

    def _setup(self: PostProcessor, configs: list[str | od.Config] | None = None) -> None:
        # set up configs
        if configs:
            self._set_configs(configs)

        # run setup hook
        self.setup_func()

    #
    # required hooks
    #

    @classmethod
    def variables(cls, func: Callable | None = None) -> None:
        """
        Decorator to wrap a function *func* that should be registered as :py:meth:`variables_func`
        and should return a sequence of multi-dimensional variables used to fill input histograms.
        The function should accept the following arguments:
            - *task*: the invoking task instance;

        The decorator does not return the wrapped function.
        """
        cls.variables_func = func

    def variables_func(self, task: law.Task) -> None:
        """
        Default variables function.
        """
        return set()

    #
    # optional hooks
    #

    @classmethod
    def setup(cls, func: Callable | None = None) -> None:
        """
        Decorator to wrap a function *func* that should be registered as :py:meth:`setup_func`.
        The function is called after the post-processor has been set up and its
        :py:attr:`config_insts` were assigned.

        The decorator does not return the wrapped function.
        """
        cls.setup_func = func

    def setup_func(self) -> None:
        """
        Default setup function.
        """
        return None

    def requires(self: PostProcessor, task: law.Task) -> Any:
        """
        Returns additional tasks that are required for the post-processor to run and whose outputs are needed.
        """
        return {}

    #
    # buisness logic methods
    #

    def run_step(self: PostProcessor, task: law.Task, step: str, inputs: dict[hist.Hist], **kwargs) -> dict[hist.Hist]:
        """
        Run the post-processing *step* on the given input histograms *inputs*, checking that all required
        inputs are provided and all declared outputs are produced.
        """
        # retrieve step configuration
        if (step_dict := self.steps.get(step, None)) is None:
            raise ValueError(f"unknown post-processing step '{step}'")

        # check inputs
        missing_inputs = {
            input_
            for input_ in (step_dict["inputs"] or [])
            if input_ not in inputs
        }
        if missing_inputs:
            missing_inputs_str = ",".join(sorted(missing_inputs))
            raise ValueError(
                f"post-processing step '{step}' missing expected inputs: {missing_inputs_str}",
            )

        # add kwargs from decorator call
        kwargs = {
            **{
                k: v
                for k, v in step_dict.items()
                if k not in {"name", "func", "inputs", "outputs"}
            },
            **kwargs,
        }

        # run post-processing step
        inputs = DotDict.wrap(inputs)
        outputs = step_dict["func"](self, task, inputs, **kwargs)

        # check outputs
        missing_outputs = {
            output
            for output in (step_dict["outputs"] or [])
            if output not in outputs
        }
        if missing_outputs:
            missing_outputs_str = ",".join(sorted(missing_outputs))
            raise ValueError(
                f"post-processing step '{step}' did not produce expected outputs: {missing_outputs_str}",
            )

        # return outputs
        return DotDict.wrap(outputs)


postprocessor = PostProcessor.postprocessor
