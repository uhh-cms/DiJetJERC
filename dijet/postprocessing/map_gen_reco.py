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
from dijet.hist_util import hist_mean_variance

hist = maybe_import("hist")
np = maybe_import("numpy")

#
# create and intialize post-processor
#

map_gen_reco = postprocessor(
    "map_gen_reco",
    #
    # user-defined parameters
    #
    # map of level and variable key to actual variable name
    variable_map={
        # gen-level variables
        "gen": {
            "alpha": "dijets_alpha",
            # variable used for binning
            "abseta": "matchedjets3_abseta",
            "pt": "genjets3_pt",
            # variable representing the raw response
            "asymmetry": "matchedjets3_pt",
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


@map_gen_reco.setup
def map_gen_reco_setup(self: PostProcessor):
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


@map_gen_reco.variables
def map_gen_reco_variables(self: PostProcessor, task: law.Task):
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

@map_gen_reco.step(
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
    import correctionlib.schemav2 as cs
    from dijet.correction_util import build_gen_reco_map
    for axis in self.rebin_cfg.keys():
        for response, h in inputs.items():
            inputs[response] = apply_rebinning_edges(h, axis, self.rebin_cfg[axis])

    if self.trim_tails_implementation == "quantiles":
        outputs = trim_tails_impl(self, task, inputs, level, **kwargs)
        h = outputs["asym.norm"][0, -1, :, :, :]
        h_mean = hist_mean_variance(h, f"{self.variable_map[level]['asymmetry']}")[:, :, "mean"]
        eta = h.axes[0].edges
        pt = h.axes[1].edges
        corr = build_gen_reco_map(pt, eta)
        for i in range(len(eta) - 1):
            for j in range(len(pt) - 1):
                corr.data.content[i].content[j] = h_mean[i, j].value if not np.isnan(h_mean[i, j].value) else 0.5 * (pt[j] + pt[j + 1])  # noqa: E501
        cset = cs.CorrectionSet(
            schema_version=2,
            description="Gen-to-reco map correction set",
            corrections=[corr],
        )
        cset_json = cset.json(exclude_unset=True)
        with open("gen_reco_map.json", "w") as fout:
            fout.write(cset_json)
        # return trim_tails_impl(self, task, inputs, level, **kwargs)
        return outputs
    elif self.trim_tails_implementation == "quantiles_CI":
        if self.trim_tails_use_max_bin:
            kwargs["use_max_bin"] = True
        return trim_tails_CI_impl(self, task, inputs, level, **kwargs)
    else:
        raise NotImplementedError(
            f"Requested trim_tails_implementation '{self.trim_tails_implementation}' is not implemented.",
        )


@map_gen_reco.step(
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


@map_gen_reco.step(
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


@map_gen_reco.step(
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


@map_gen_reco.step(
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


map_gen_reco_fine = map_gen_reco.derive(
    "map_gen_reco_fine",
    cls_dict={
        "variable_map": {
            "gen": {
                "alpha": "dijets_alpha",
                "abseta": "matchedjets3_abseta",
                "pt": "genjets3_pt",
                "asymmetry": "matchedjets3_pt_fine",
            },
        },
    },
)
