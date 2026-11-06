# coding: utf-8

"""
Postprocessor for estimating JER from dijet balance and alpha extrapolation.
"""

from __future__ import annotations

import law

from columnflow.util import maybe_import

from dijet.hist_util import apply_rebinning_edges

from dijet.postprocessing import PostProcessor, postprocessor
from dijet.postprocessing.base import (
    trim_tails_impl,
    trim_tails_CI_impl,
    extract_width_impl,
    extrapolate_width_impl,
    calc_jer_impl,
    calc_sf_impl,
)

hist = maybe_import("hist")

#
# create and intialize post-processor
#

mctruth = postprocessor(
    "mctruth",
    #
    # user-defined parameters
    #
    # map of level and variable key to actual variable name
    variable_map={
        # gen-level variables
        "gen": {
            "alpha": "dijets_alpha",
            # variable used for binning
            "abseta": "genjet1_abseta",
            "pt": "genjet1_pt",
            # variable representing the raw response
            "asymmetry": "mc_truth_response",
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
    trim_tails_asymmetry_rebin_factor=1,
    # total fraction of response distribution to remove symetrically from both
    # ends in order to cut off non-Gaussian tails
    trim_tails_fraction=0.03,
    # method to find the thresholds where to trim the non-Gaussian tails
    trim_tails_implementation="quantiles",
    # wether to start from mean or the bin with largest value
    trim_tails_use_max_bin=False,
    # method to use for extracting width
    extract_width_method="empirical",
    # name of response used for JER computation
    calc_jer_main_response="asym",
    # whether to use extrapolated results (True)
    # or take the most inclusive bin (False)
    calc_jer_use_extrapolation=False,
    # whether to subtract the particle-level imbalance
    # (gen-level resolution) when calculating JER
    calc_jer_subtract_pli=False,
    # upper |eta| value until which to average
    # SM JER to use as a reference for the FE method
    calc_jer_max_abseta_sm_ref=1.131,
    # variables to rebin corser than the default binning
    rebin_cfg={},
)


@mctruth.setup
def mctruth_setup(self: PostProcessor):
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


@mctruth.variables
def mctruth_variables(self: PostProcessor, task: law.Task):
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

@mctruth.step(
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
    for axis in self.rebin_cfg.keys():
        for response, h in inputs.items():
            inputs[response] = apply_rebinning_edges(h, axis, self.rebin_cfg[axis])

    if self.trim_tails_implementation == "quantiles":
        return trim_tails_impl(self, task, inputs, level, **kwargs)
    elif self.trim_tails_implementation == "quantiles_CI":
        if self.trim_tails_use_max_bin:
            kwargs["use_max_bin"] = True
        return trim_tails_CI_impl(self, task, inputs, level, **kwargs)
    else:
        raise NotImplementedError(
            f"Requested trim_tails_implementation '{self.trim_tails_implementation}' is not implemented.",
        )


@mctruth.step(
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


@mctruth.step(
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


@mctruth.step(
    name="calc_jer",
    inputs={
        "asym.width.gen",
        "asym.extrapolation.gen.intercepts",
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
    gen_only: bool = False,
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
    return calc_jer_impl(self, task, inputs, gen_only=gen_only, **kwargs)


@mctruth.step(
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


mctruth_87CI = mctruth.derive(
    "mctruth_87CI",
    cls_dict={
        "variable_map": {
            "gen": {
                "alpha": "dijets_alpha",
                "abseta": "genjets_abseta",
                "pt": "genjets_pt",
                "asymmetry": "mc_truth_responses",
            },
        },
        "trim_tails_fraction": 0.13,
        "trim_tails_implementation": "quantiles_CI",
    },
)

mctruth_87CI_a1 = mctruth.derive(
    "mctruth_87CI_a1",
    cls_dict={
        "variable_map": {
            "gen": {
                "alpha": "dijets_alpha",
                "abseta": "genjets_abseta",
                "pt": "genjets_pt",
                "asymmetry": "mc_truth_responses",
            },
        },
        "trim_tails_fraction": 0.13,
        "trim_tails_implementation": "quantiles_CI",
    },
)
mctruth_97CI_a1 = mctruth.derive(
    "mctruth_97CI_a1",
    cls_dict={
        "variable_map": {
            "gen": {
                "alpha": "dijets_alpha",
                "abseta": "genjets_abseta",
                "pt": "genjets_pt",
                "asymmetry": "mc_truth_responses",
            },
        },
        "trim_tails_fraction": 0.03,
        "trim_tails_implementation": "quantiles_CI",
    },
)
mctruth3j_97CI_a1 = mctruth.derive(
    "mctruth3j_97CI_a1",
    cls_dict={
        "variable_map": {
            "gen": {
                "alpha": "dijets_alpha",
                "abseta": "genjets3_abseta",
                "pt": "genjets3_pt",
                "asymmetry": "mc_truth_responses3",
            },
        },
        "trim_tails_fraction": 0.03,
        "trim_tails_implementation": "quantiles_CI",
    },
)
mctruth1j_97CI_a1 = mctruth.derive(
    "mctruth1j_97CI_a1",
    cls_dict={
        "variable_map": {
            "gen": {
                "alpha": "dijets_alpha",
                "abseta": "genjet1_abseta",
                "pt": "genjet1_pt",
                "asymmetry": "mc_truth_response",
            },
        },
        "trim_tails_fraction": 0.03,
        "trim_tails_implementation": "quantiles_CI",
    },
)

mctruth_87CI_a1_merged = mctruth.derive(
    "mctruth_87CI_a1_merged",
    cls_dict={
        "variable_map": {
            "gen": {
                "alpha": "dijets_alpha",
                "abseta": "genjets_abseta",
                "pt": "genjets_pt",
                "asymmetry": "mc_truth_responses",
            },
        },
        "trim_tails_fraction": 0.13,
        "trim_tails_implementation": "quantiles_CI",
        "rebin_cfg": {
            "genjets_abseta": [0, 1.305, 2.500, 5.191],
        },
    },
)

mctruth_97CI_a1_merged = mctruth.derive(
    "mctruth_97CI_a1_merged",
    cls_dict={
        "variable_map": {
            "gen": {
                "alpha": "dijets_alpha",
                "abseta": "genjets_abseta",
                "pt": "genjets_pt",
                "asymmetry": "mc_truth_responses",
            },
        },
        "trim_tails_fraction": 0.03,
        "trim_tails_implementation": "quantiles_CI",
        "rebin_cfg": {
            "genjets_abseta": [0, 1.305, 2.500, 5.191],
        },
    },
)

mctruth_87CI_merged = mctruth.derive(
    "mctruth_87CI_merged",
    cls_dict={
        "variable_map": {
            "gen": {
                "alpha": "dijets_alpha",
                "abseta": "genjets_abseta",
                "pt": "genjets_pt",
                "asymmetry": "mc_truth_responses",
            },
        },
        "trim_tails_fraction": 0.13,
        "trim_tails_implementation": "quantiles_CI",
        "rebin_cfg": {
            "genjets_abseta": [0, 1.305, 2.500, 5.191],
        },
    },
)


mctruth_97CI = mctruth.derive(
    "mctruth_97CI",
    cls_dict={
        "variable_map": {
            "gen": {
                "alpha": "dijets_alpha",
                "abseta": "genjets_abseta",
                "pt": "genjets_pt",
                "asymmetry": "mc_truth_responses",
            },
        },
        "trim_tails_fraction": 0.03,
        "trim_tails_implementation": "quantiles_CI",
    },
)


mctruth_97CI_merged = mctruth.derive(
    "mctruth_97CI_merged",
    cls_dict={
        "variable_map": {
            "gen": {
                "alpha": "dijets_alpha",
                "abseta": "genjets_abseta",
                "pt": "genjets_pt",
                "asymmetry": "mc_truth_responses",
            },
        },
        "trim_tails_fraction": 0.03,
        "trim_tails_implementation": "quantiles_CI",
        "rebin_cfg": {
            "genjets_abseta": [0, 1.305, 2.500, 5.191],
        },
    },
)


mctruth_87CI_reco_a1 = mctruth.derive(
    "mctruth_87CI_reco_a1",
    cls_dict={
        "variable_map": {
            "gen": {
                "alpha": "dijets_alpha",
                "abseta": "matchedjets_abseta",
                "pt": "matchedjets_pt",
                "asymmetry": "mc_truth_responses",
            },
        },
        "trim_tails_fraction": 0.13,
        "trim_tails_implementation": "quantiles_CI",
    },
)

mctruth_97CI_reco_a1 = mctruth.derive(
    "mctruth_97CI_reco_a1",
    cls_dict={
        "variable_map": {
            "gen": {
                "alpha": "dijets_alpha",
                "abseta": "matchedjets_abseta",
                "pt": "matchedjets_pt",
                "asymmetry": "mc_truth_responses",
            },
        },
        "trim_tails_fraction": 0.03,
        "trim_tails_implementation": "quantiles_CI",
    },
)

mctruth_97CI_reco3_a1 = mctruth.derive(
    "mctruth_97CI_reco3_a1",
    cls_dict={
        "variable_map": {
            "gen": {
                "alpha": "dijets_alpha",
                "abseta": "matchedjets3_abseta",
                "pt": "matchedjets3_pt",
                "asymmetry": "mc_truth_responses3",
            },
        },
        "trim_tails_fraction": 0.03,
        "trim_tails_implementation": "quantiles_CI",
    },
)
mctruth_97CI_mixed3_a1 = mctruth.derive(
    "mctruth_97CI_mixed3_a1",
    cls_dict={
        "variable_map": {
            "gen": {
                "alpha": "dijets_alpha",
                "abseta": "matchedjets3_abseta",
                "pt": "genjets3_pt",
                "asymmetry": "mc_truth_responses3",
            },
        },
        "trim_tails_fraction": 0.03,
        "trim_tails_implementation": "quantiles_CI",
    },
)
mctruth_97CI_reco3_a1_merged = mctruth.derive(
    "mctruth_97CI_reco3_a1_merged",
    cls_dict={
        "variable_map": {
            "gen": {
                "alpha": "dijets_alpha",
                "abseta": "matchedjets3_abseta",
                "pt": "matchedjets3_pt",
                "asymmetry": "mc_truth_responses3",
            },
        },
        "trim_tails_fraction": 0.03,
        "trim_tails_implementation": "quantiles_CI",
        "rebin_cfg": {
            "matchedjets3_abseta": [0, 1.305, 2.500, 5.191],
        },
    },
)

mctruth_87CI_reco_merged = mctruth.derive(
    "mctruth_87CI_reco_merged",
    cls_dict={
        "variable_map": {
            "gen": {
                "alpha": "dijets_alpha",
                "abseta": "matchedjets_abseta",
                "pt": "matchedjets_pt",
                "asymmetry": "mc_truth_responses",
            },
        },
        "trim_tails_fraction": 0.13,
        "trim_tails_implementation": "quantiles_CI",
        "rebin_cfg": {
            "matchedjets_abseta": [0, 1.305, 2.500, 5.191],
        },
    },
)


#
# TODO: clean up (delete)
#

mctruth_closure = mctruth.derive(
    "mctruth_closure",
    cls_dict={
        "variable_map": {
            "gen": {
                "alpha": "dijets_alpha",
                "abseta": "genjets_abseta",
                "pt": "genjets_pt",
                "asymmetry": "mc_truth_responses",
            },
        },
        "trim_tails_fraction": 0.13,
        "trim_tails_implementation": "quantiles_CI",
    },
)


mctruth_recobins = mctruth.derive(
    "mctruth_recobins",
    cls_dict={
        "variable_map": {
            "gen": {
                "alpha": "dijets_alpha",
                "abseta": "matchedjet_abseta",
                "pt": "matchedjet_pt",
                "asymmetry": "mc_truth_response",
            },
        },
        "trim_tails_fraction": 0.03,
    },
)


mctruth_recobins2 = mctruth.derive(
    "mctruth_recobins2",
    cls_dict={
        "variable_map": {
            "gen": {
                "alpha": "dijets_alpha",
                "abseta": "matchedjets_abseta",
                "pt": "matchedjets_pt",
                "asymmetry": "mc_truth_responses",
            },
        },
        "trim_tails_fraction": 0.03,
    },
)


mctruth_corsebin = mctruth.derive(
    "mctruth_corsebin",
    cls_dict={
        "variable_map": {
            "gen": {
                "alpha": "dijets_alpha",
                "abseta": "matchedjets_abseta",
                "pt": "matchedjets_pt",
                "asymmetry": "mc_truth_responses",
            },
        },
        "trim_tails_fraction": 0.13,
        "rebin_cfg": {
            "matchedjets_abseta": [0, 1.305, 2.500, 5.191],
        },
    },
)


mctruth_2GJ_merged_87 = mctruth.derive(
    "mctruth_2GJ_merged_87",
    cls_dict={
        "variable_map": {
            "gen": {
                "alpha": "dijets_alpha",
                "abseta": "genjets_abseta",
                "pt": "genjets_pt",
                "asymmetry": "mc_truth_responses",
            },
        },
        "trim_tails_fraction": 0.13,
        "rebin_cfg": {
            "genjets_abseta": [0, 1.305, 2.500, 5.191],
        },
    },
)


mctruth_closure_gauss = mctruth.derive(
    "mctruth_closure_gauss",
    cls_dict={
        "variable_map": {
            "gen": {
                "alpha": "dijets_alpha",
                "abseta": "genjets_abseta",
                "pt": "genjets_pt",
                "asymmetry": "mc_truth_responses",
            },
        },
        "trim_tails_fraction": 0.13,
        "trim_tails_implementation": "quantiles_CI",
        "extract_width_method": "gaussian_fit",
    },
)


mctruth_twojets = mctruth.derive(
    "mctruth_twojets",
    cls_dict={
        "variable_map": {
            "gen": {
                "alpha": "dijets_alpha",
                "abseta": "genjets_abseta",
                "pt": "genjets_pt",
                "asymmetry": "mc_truth_responses",
            },
        },
        "trim_tails_fraction": 0.13,
    },
)


mctruth_87 = mctruth.derive(
    "mctruth_87",
    cls_dict={
        "trim_tails_fraction": 0.13,
    },
)


mctruth_87CI = mctruth.derive(
    "mctruth_87CI",
    cls_dict={
        "trim_tails_fraction": 0.13,
        "trim_tails_implementation": "quantiles_CI",
    },
)


mctruth_97CI = mctruth.derive(
    "mctruth_97CI",
    cls_dict={
        "trim_tails_fraction": 0.03,
        "trim_tails_implementation": "quantiles_CI",
    },
)


mctruth_trimCI = mctruth.derive(
    "mctruth_trimCI",
    cls_dict={
        "trim_tails_fraction": 0.13,
        "trim_tails_implementation": "quantiles_CI",
    },
)

mctruth_test = mctruth.derive(
    "mctruth_test",
    cls_dict={
        "trim_tails_fraction": 0.13,
    },
)


mctruth_gaussian_fit = mctruth.derive(
    "mctruth_gaussian_fit",
    cls_dict={
        "extract_width_method": "gaussian_fit",
    },
)


mctruth_no_subtract_pli = mctruth.derive(
    "mctruth_no_subtract_pli",
    cls_dict={
        "calc_jer_subtract_pli": False,
    },
)
