# coding: utf-8

"""
Postprocessor for estimating JER from dijet balance and alpha extrapolation.
"""

from __future__ import annotations

import law

from columnflow.util import maybe_import

from dijet.postprocessing import PostProcessor, postprocessor
from dijet.postprocessing.base import (
    trim_tails_impl,
    extract_width_impl,
    extrapolate_width_impl,
    calc_jer_impl,
    calc_sf_impl,
)

hist = maybe_import("hist")

#
# create and intialize post-processor
#

dijet_balance = postprocessor(
    "dijet_balance",
    #
    # user-defined parameters
    #
    # map of level and variable key to actual variable name
    variable_map={
        # reco-level variables
        "reco": {
            # variable used for extrapolating raw response
            "alpha": "dijets_alpha",
            # variable used for binning
            "abseta": "probejet_abseta",
            "pt": "dijets_pt_avg",
            # variable representing the raw response
            "asymmetry": "dijets_asymmetry",
        },
        # gen-level variables
        "gen": {
            # variable used for extrapolating raw response
            "alpha": "dijets_alpha",  # TODO: change to alpha_gen?
            # variable used for binning
            "abseta": "probejet_abseta_gen",
            "pt": "dijets_pt_avg_gen",
            # variable representing the raw response
            "asymmetry": "dijets_asymmetry_gen",
        },
    },
    # configuration for response quantities
    response_cfg={
        # responses derived directly from input histograms
        "asym": {
            "response_var_key": "asymmetry",
        },
    },
    # key of variable used for extrapolation (not extrapolation or response itself)
    extrapolation_var_key="alpha",
    # keys of variables used for binning (not extrapolation or response itself)
    binning_var_keys=("abseta", "pt"),
    # rebinning factor to apply to asymmetry axis
    trim_tails_asymmetry_rebin_factor=2,
    # total fraction of response distribution to remove symetrically from both
    # ends in order to cut off non-Gaussian tails
    trim_tails_fraction=0.03,
    # method to use for extracting width
    extract_width_method="empirical",
    # name of response used for JER computation
    calc_jer_main_response="asym",
    # whether to use extrapolated results (True)
    # or take the most inclusive bin (False)
    calc_jer_use_extrapolation=True,
    # whether to subtract the particle-level imbalance
    # (gen-level resolution) when calculating JER
    calc_jer_subtract_pli=True,
    # upper |eta| value until which to average
    # SM JER to use as a reference for the FE method
    calc_jer_max_abseta_sm_ref=1.131,
)


@dijet_balance.setup
def dijet_balance_setup(self: PostProcessor):
    """
    Set up post-processor. Populate *self.responses* dict with information about
    what to do for each response.
    """
    # set inputs
    self.responses = {}
    for response_key, response_spec in self.response_cfg.items():
        # validate config
        if not (response_var_key := response_spec.get("response_var_key", None)):
            raise ValueError(
                f"response '{response_key}' does not define mandatory "
                "'response_var_key' in response_cfg",
            )

        # keys of all input variables
        all_var_keys = (
            self.extrapolation_var_key,
        ) + self.binning_var_keys + (
            response_var_key,
        )

        # helper function for resolving actual
        # variable names per level
        def resolve_vars(var_keys, level):
            return [
                self.variable_map[level][var_key]
                for var_key in var_keys
            ]

        # construct input specification (one dict
        # per response)
        self.responses[response_key] = dict(
            response_spec,
            **{
                # key and resolved names of single response variable
                "response_var_key": response_var_key,
                "response_vars": {
                    level: resolve_vars([response_var_key], level)[0]
                    for level in self.variable_map
                },

                # key and resolved names of all variables
                "all_var_keys": all_var_keys,
                "all_vars": {
                    level: resolve_vars(all_var_keys, level)
                    for level in self.variable_map
                },

                # multi-dimensional input variables to pass on to
                # cf.MergeHistograms
                "hist_vars": {
                    level: "-".join(resolve_vars(all_var_keys, level))
                    for level in self.variable_map
                },
            },
        )


@dijet_balance.variables
def dijet_balance_variables(self: PostProcessor, task: law.Task):
    """
    Return sequence of multidimensional variables used for
    filling input histograms.
    """
    # compute multi-dimensional input variables
    # from input configuration
    return {
        level: {
            input_key: input_spec["hist_vars"][level]
            for input_key, input_spec in self.responses.items()
            if not input_spec.get("derived", False)
        }
        for level in self.variable_map
    }


#
# register post-processing steps
#

@dijet_balance.step(
    name="trim_tails",
    inputs={"asym.dist"},
    outputs={
        "asym.{norm,cut,nevt,cut_edges}",
    },
)
def trim_tails(
    self: PostProcessor,
    task: law.Task,
    inputs: dict[hist.Hist],
    # user-defined parameters
    level: str,
    **kwargs,
) -> dict[hist.Hist]:
    """
    Trim the (possibly non-Gaussian) tails of the response distributions.

    Inputs
    ------
    * ``dist``: a multi-dimensional histogram containing the raw response distribution

    Outputs
    -------
    * ``norm``: a multi-dimensional histogram containing the normalized response distributions
               before trimming non-Gaussian tails
    * ``cut``: a multi-dimensional histogram containing the normalized response distributions
               after trimming non-Gaussian tails
    * ``nevt``: a multi-dimensional histogram containing the event yields (integral over response)
    * ``cut_edges``: a dict of histograms ``low`` and ``up`` containing the lower and upper bound
                     of the response interval after trimming non-Gaussian tails (for plotting)
    """
    return trim_tails_impl(self, task, inputs, level, **kwargs)


@dijet_balance.step(
    name="extract_width",
    inputs={
        "asym.{cut,nevt}",
    },
    outputs={
        "asym.width",
    },
    # TODO: separate out "trim_tails", "count_events" and "calc_quantiles" ?
)
def extract_width(
    self: PostProcessor,
    task: law.Task,
    inputs: dict[hist.Hist],
    # user-defined parameters
    level: str,
    **kwargs,
) -> dict[hist.Hist]:
    """
    Extract the width of the response distributions.

    Inputs
    ------
    * ``cut``: a multi-dimensional histogram containing the trimmed response distributions
    * ``nevt``: a multi-dimensional histogram containing the event yields (integral over response)

    Outputs
    -------
    * ``width``: the width of the response distributions
    """
    return extract_width_impl(self, task, inputs, level, **kwargs)


@dijet_balance.step(
    name="extrapolate_width",
    inputs={
        "asym.{width,nevt}",
    },
    outputs={
        "asym.extrapolation",
    },
)
def extrapolate_width(
    self: PostProcessor,
    task: law.Task,
    inputs: dict[hist.Hist],
    # user-defined parameters
    level: str,
    **kwargs,
) -> dict[hist.Hist]:
    """
    Extrapolate the width of the response distributions to alpha=0.

    Inputs
    ------
    * ``width``: a multi-dimensional histogram containing the widths of the response distributions
    * ``nevt``: a multi-dimensional histogram containing the event yields (integral over response)

    Outputs
    -------
    * ``extrapolation``: the extrapolated width of the response distribution, calculated via a linear
                         fit with correlations
    """
    return extrapolate_width_impl(self, task, inputs, level, **kwargs)


@dijet_balance.step(
    name="calc_jer",
    inputs={
        "asym.width.{reco,gen}",
        "asym.extrapolation.{reco,gen}.intercepts",
    },
    outputs={
        "asym.jer",
    },
)
def calc_jer(
    self: PostProcessor,
    task: law.Task,
    inputs: dict[hist.Hist],
    # user-defined parameters
    **kwargs,
) -> dict[hist.Hist]:
    """
    Compute the JER from the extracted widths. The SM JER is computed directly from the width,
    while the FE JER uses the SM JER in the first few |eta| bins as a reference.

    Optionally, the particle-level imbalance (i.e. the extrapolated gen-level response width)
    is subtracted in quadrature from the reconstructed-level response width before computing
    the JER

    Inputs
    ------
    * ``width.{reco,gen}``: a multi-dimensional histogram containing the extrapolated
      widths on reconstruction- and generator-level, respectively

    Outputs
    -------
    * ``jer``: the JER, calculated from the extrapolated widths (TODO: document formulae)
    """
    return calc_jer_impl(self, task, inputs, **kwargs)


@dijet_balance.step(
    name="calc_sf",
    inputs={
        "asym.jer.{data,mc}",
    },
    outputs={
        "asym.jer_sf",
    },
)
def calc_sf(
    self: PostProcessor,
    task: law.Task,
    inputs: dict[hist.Hist],
    # user-defined parameters
    **kwargs,
) -> dict[hist.Hist]:
    """
    Calculate the jet energy resolution scale factor from the JER in data and MC.

    Inputs
    ------
    * ``jer.data``: a multi-dimensional histogram containing the JER in data
    * ``jer.mc``: a multi-dimensional histogram containing the JER in MC

    Outputs
    -------
    * ``jer_sf``: the JER scale factor, calculated as the ratio of JERs in data and MC
    """
    return calc_sf_impl(self, task, inputs, **kwargs)


dijet_balance_gaussian_fit = dijet_balance.derive(
    "dijet_balance_gaussian_fit",
    cls_dict={
        "extract_width_method": "gaussian_fit",
    },
)


dijet_balance_no_subtract_pli = dijet_balance.derive(
    "dijet_balance_no_subtract_pli",
    cls_dict={
        "calc_jer_subtract_pli": False,
    },
)
