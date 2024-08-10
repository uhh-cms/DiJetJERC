# coding: utf-8

"""
Custom mixins for adding parameters/methods specific to
dijet analysis.
"""

import luigi
import law
import order as od

from columnflow.tasks.framework.mixins import VariablesMixin


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
