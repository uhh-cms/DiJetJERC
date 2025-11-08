# coding: utf-8

"""
Postprocessors for JER.
"""

from __future__ import annotations

import itertools as it
import fnmatch

import law

from columnflow.util import maybe_import
from columnflow.types import Callable

from dijet.postprocessing import PostProcessor, postprocessor

from dijet.hist_util import hist_mean_variance, hist_fit_gaussian, hist_sub_in_quadrature
from dijet.fit_util import CorrelatedFit
from dijet.util import deflate_dict, inflate_dict

np = maybe_import("numpy")
ak = maybe_import("awkward")
hist = maybe_import("hist")


#
# utility functions
#

def check_hist_axes(hists: dict, checks: dict[Callable]):
    """
    Check that all *inputs* contain the required axes, specified via *checks*. The
    *checks* should be a dictionary mapping patterns to functions that return
    a sequence of strings indicating the variable keys of the required axes. A ``ValueError``
    is raised if one or more axes are missing.
    """

    flat_hists = deflate_dict(hists)

    def _check_input_single(key: str, input_hist: hist.Hist, axis_names: set[str]):
        # compare to existing axes
        existing_axis_names = {a.name for a in input_hist.axes}
        missing_axes = set(axis_names) - existing_axis_names
        if missing_axes:
            missing_axes_str = ",".join(sorted(missing_axes))
            existing_axis_names_str = ",".join(sorted(existing_axis_names))
            raise ValueError(
                f"histogram for input '{key}' is missing axes; "
                f"expected: {missing_axes_str}, available: {existing_axis_names_str}",
            )

    for key, hist_ in flat_hists.items():
        for pattern, axis_names in checks.items():
            if fnmatch.fnmatch(key, pattern):
                _check_input_single(key, hist_, axis_names)


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
    # inflate inputs
    inputs = inflate_dict(inputs)

    # initialize storage for outputs
    outputs = {}

    # use correct level for variable lookup
    variable_map = self.variable_map[level]

    # loop through responses
    for response_key, response_cfg in self.responses.items():
        # get name of response variable for this histogram
        response_var_key = response_cfg["response_var_key"]

        # retrieve inputs for response
        r_inputs = inputs[response_key]
        r_outputs = outputs[response_key] = {}

        # check inputs for response
        all_vars = set(response_cfg["all_vars"][level])
        check_hist_axes(
            hists=r_inputs,
            checks={
                "*": all_vars,
            },
        )

        # copy input histogram
        h_in = r_inputs["dist"].copy(deep=False)

        # rebin response
        h_in = h_in[{
            variable_map[response_var_key]: hist.rebin(
                self.trim_tails_asymmetry_rebin_factor,
            ),
        }]

        # resolve axis names and indices
        axis_names = [a.name for a in h_in.axes]
        axis_indices = {
            var_key: axis_names.index(variable_map[var_key])
            for var_key in (self.extrapolation_var_key, response_var_key)
        }

        # work on view of histogram data
        v_in = h_in.view()

        # replace histogram contents with cumulative sum over
        # bins in the extrapolation variable (typically 'alpha')
        v_in.value = np.apply_along_axis(np.cumsum, axis=axis_indices[self.extrapolation_var_key], arr=v_in.value)
        v_in.variance = np.apply_along_axis(np.cumsum, axis=axis_indices[self.extrapolation_var_key], arr=v_in.variance)

        # get total event yield from integral over asymmetry distrivution
        # (note: we don't include over-/underflow bins here)
        integral = h_in.values().sum(axis=axis_indices[response_var_key], keepdims=True)
        integral_var = h_in.variances().sum(axis=axis_indices[response_var_key], keepdims=True)

        # store event yield as histogram in output
        h_nevt = h_in.copy()
        h_nevt = h_nevt[{variable_map[response_var_key]: sum}]
        h_nevt.view().value = np.squeeze(integral)
        h_nevt.view().variance = np.squeeze(integral_var)
        r_outputs["nevt"] = h_nevt.copy(deep=False)

        # normalize histogram to integral over asymmetry
        v_in.value = v_in.value / integral
        v_in.variance = v_in.variance / integral**2

        # store full asymmetry distribution (including non-Gaussian tails)
        r_outputs["norm"] = h_in.copy(deep=False)

        #
        # next, trim non-Gaussian tails
        #

        # get cumulative response distribution
        fraction = np.cumsum(v_in.value, axis=axis_indices[response_var_key])
        # TODO: last fraction should be 1 -> check

        # compute lower and upper quantiles for trimming
        quantile_lo = self.trim_tails_fraction / 2
        quantile_up = 1.0 - quantile_lo

        # find index of bins corresponding to quantiles
        ind_lo = np.apply_along_axis(np.searchsorted, axis_indices[response_var_key], fraction, quantile_lo, side="left")
        ind_up = np.apply_along_axis(np.searchsorted, axis_indices[response_var_key], fraction, quantile_up, side="right")

        # add back reduced dimension
        ind_lo = np.expand_dims(ind_lo, axis=axis_indices[response_var_key])
        ind_up = np.expand_dims(ind_up, axis=axis_indices[response_var_key])

        # obtain bin edges corresponding to requested quantiles
        asym_edges = h_in.axes[variable_map[response_var_key]].edges
        asym_edges_lo = asym_edges[ind_lo]
        asym_edges_up = asym_edges[ind_up + 1]  # upper edge -> add 1

        # store bin edges in histogram format
        h_asym_edges_lo = h_in.copy()[{variable_map[response_var_key]: sum}]
        h_asym_edges_up = h_asym_edges_lo.copy()
        h_asym_edges_lo.view().value = np.squeeze(asym_edges_lo)
        h_asym_edges_lo.view().variance[:] = 0.0
        h_asym_edges_up.view().value = np.squeeze(asym_edges_up)
        h_asym_edges_up.view().variance[:] = 0.0
        r_outputs["cut_edges"] = {
            "low": h_asym_edges_lo.copy(),
            "up": h_asym_edges_up.copy(),
        }

        # compute mask to select tail bins
        asym_centers = h_in.axes[variable_map[response_var_key]].centers
        # asym_centers = asym_centers.reshape(1, 1, 1, 1, -1)  # TODO: not hard-coded
        asym_centers_shape = (1,) * (asym_edges_lo.ndim - 1) + (-1,)
        asym_centers = asym_centers.reshape(*asym_centers_shape)
        # asym_centers = np.expand_dims(asym_centers, axis=axis_indices[response_var_key])
        mask = (asym_centers > asym_edges_lo) & (asym_centers < asym_edges_up)

        # set tail bins to zero for trimming
        v_in.value = np.where(mask, v_in.value, 0)
        v_in.variance = np.where(mask, v_in.variance, 0)

        # store asymmetry with trimmed non-Gaussian tails
        r_outputs["cut"] = h_in.copy(deep=False)

        # check outputs
        check_hist_axes(
            hists=r_outputs,
            checks={
                "norm": all_vars,
                "cut": all_vars,
                "nevt": all_vars - {response_cfg["response_vars"][level]},
                "cut_edges": all_vars - {response_cfg["response_vars"][level]},
            },
        )

    # deflate outputs
    outputs = deflate_dict(outputs, max_depth=2)

    # return outputs
    return outputs


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
    # inflate inputs
    inputs = inflate_dict(inputs)

    # initialize storage for outputs
    outputs = {}

    # use correct level for variable lookup
    variable_map = self.variable_map[level]

    # loop through responses
    for response_key, response_cfg in self.responses.items():
        # skip derived responses
        if response_cfg.get("derived", False):
            continue

        # get name of response variable for this histogram
        response_var_key = response_cfg["response_var_key"]

        # retrieve inputs for response
        r_inputs = inputs[response_key]
        r_outputs = outputs[response_key] = {}

        # check inputs for response
        all_vars = set(response_cfg["all_vars"][level])
        check_hist_axes(
            hists=r_inputs,
            checks={
                "cut": all_vars,
                "nevt": all_vars - {response_cfg["response_vars"][level]},
            },
        )

        # copy input histograms
        h_dist = r_inputs["cut"].copy(deep=False)
        h_nevt = r_inputs["nevt"].copy(deep=False)

        # resolve axis names and indices
        axis_names = [a.name for a in h_dist.axes]

        #
        # start main processing
        #

        # check that asymmetry axis is last (TODO: remove this limitation)
        if axis_names[-1] != variable_map[response_var_key]:
            raise RuntimeError(
                "internal error: asymmetry axis must come last",
            )

        #
        # method choice: Gaussian fit or empirical standard deviation
        #

        if self.extract_width_method == "gaussian_fit":
            # fit Gaussian to input histograms along asymmetry axis
            h_fit_gaussian = hist_fit_gaussian(
                h_dist,
                axis=variable_map[response_var_key],
            )
            h_stds_fit = h_fit_gaussian[..., "sigma"]
            v_stds_fit = h_stds_fit.view()
            v_stds_fit.value = np.nan_to_num(v_stds_fit.value, nan=0.0)
            v_stds_fit.variance = np.nan_to_num(v_stds_fit.variance, nan=0.0)

            # use Gaussian fit results as widths
            h_stds = h_stds_fit

        elif self.extract_width_method == "empirical":
            # compute mean and variance of input histograms along asymmetry axis
            h_mean_variance = hist_mean_variance(
                h_dist,
                axis=variable_map[response_var_key],
            )

            # calculate standard deviation and its errors
            # note: the error on standard deviation is analogous to the
            # implementation in ROOT::TH1:
            # https://root.cern/doc/v630/TH1_8cxx_source.html#l07520
            h_stds_emp = h_mean_variance[..., "variance"]
            v_stds_emp = h_stds_emp.view()
            with np.errstate(invalid="ignore"):
                v_stds_emp.variance = v_stds_emp.value / (2 * h_nevt.values())
                v_stds_emp.value = np.nan_to_num(np.sqrt(v_stds_emp.value), nan=0.0)

            # use empirical standard deviations as widths
            h_stds = h_stds_emp

        else:
            raise ValueError(
                f"invalid width extraction method "
                f"'{self.extract_width_method}', expected one of: "
                f"gaussian_fit,empirical",
            )

        # Store alphas here to get alpha up to 1
        # In the next stepts only alpha<0.3 needed; avoid slicing from there
        r_outputs["width"] = h_stds

        # check outputs
        check_hist_axes(
            hists=r_outputs,
            checks={
                "width": all_vars - {response_cfg["response_vars"][level]},
            },
        )

    # deflate outputs
    outputs = deflate_dict(outputs, max_depth=2)

    # return outputs
    return outputs


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
    # inflate inputs
    inputs = inflate_dict(inputs)

    # initialize storage for outputs
    outputs = {}

    # use correct level for variable lookup
    variable_map = self.variable_map[level]

    # loop through responses
    for response_key, response_cfg in self.responses.items():

        # retrieve inputs for response
        r_inputs = inputs[response_key]
        r_outputs = outputs[response_key] = {}

        # check inputs for response
        all_vars = set(response_cfg["all_vars"][level])
        check_hist_axes(
            hists=r_inputs,
            checks={
                "*": set(
                    all_vars -
                    {response_cfg["response_vars"][level]},
                ),
            },
        )

        # copy input histograms
        h_stds = r_inputs["width"].copy(deep=False)
        h_nevt = r_inputs["nevt"].copy(deep=False)

        # Get max alpha for fit; usually 0.3
        amax = 0.3  # TODO: define in config
        h_stds = h_stds[{variable_map[self.extrapolation_var_key]: slice(0, hist.loc(amax))}]
        h_nevt = h_nevt[{variable_map[self.extrapolation_var_key]: slice(0, hist.loc(amax))}]
        # exclude 0, the first bin, from alpha edges
        alphas = h_stds.axes[variable_map[self.extrapolation_var_key]].edges[1:]

        # TODO: More efficient procedure than for loop?
        #       - Idea: Array with same shape but with tuple (width, error) as entry
        binning_variables = [variable_map[bv_key] for bv_key in self.binning_var_keys]
        n_bins = [
            len(h_stds.axes[bv].centers)
            for bv in binning_variables
        ]
        n_methods = len(h_stds.axes["category"].centers)  # only length
        inter = h_stds.copy().values()
        inter = inter[:, :2, ...]  # keep first two entries
        slope = h_stds.copy().values()
        slope = slope[:, :2, ...]  # keep first two entries
        for m, *bv_indices in it.product(
            range(n_methods),
            *[range(n) for n in n_bins],
        ):
            h_slice = {
                "category": m,
            }
            h_slice.update({
                bv: bv_index
                for bv, bv_index in zip(
                    binning_variables,
                    bv_indices,
                )
            })
            tmp = h_stds[h_slice]
            tmp_evts = h_nevt[h_slice]
            coeff, err = CorrelatedFit.get_correlated_fit(
                wmax=alphas,
                std=tmp.values(),
                nevts=tmp_evts.values(),
            )
            inter[(m, slice(None), *bv_indices)] = [coeff[1], err[1]]
            slope[(m, slice(None), *bv_indices)] = [coeff[0], err[0]]

        # NOTE: store fits into hist.
        h_intercepts = h_stds.copy()
        # Remove axis for alpha for histogram
        h_intercepts = h_intercepts[{variable_map[self.extrapolation_var_key]: sum}]
        # y intercept of fit (x=0)
        h_intercepts.view().value = inter[:, 0, ...]
        # Errors temporarly used; Later get:
        # Error on fit from fit function (how?) or new method with three fits
        h_intercepts.view().variance = inter[:, 1, ...]

        h_slopes = h_intercepts.copy()
        # Slope of fit stored in index 1
        h_slopes.view().value = slope[:, 0, ...]
        # Only stored for plotting, no defined error
        h_slopes.view().variance = slope[:, 1, ...]

        # store y-intercepts and slopes from linear fit
        # in identically-structured histograms
        r_outputs["extrapolation"] = {
            "intercepts": h_intercepts,
            "slopes": h_slopes,
        }

        # check outputs
        check_hist_axes(
            hists=r_outputs,
            checks={
                "extrapolation.*": set(
                    all_vars -
                    {
                        response_cfg["response_vars"][level],
                        variable_map[self.extrapolation_var_key],
                    },
                ),
            },
        )

    # deflate outputs
    outputs = deflate_dict(outputs, max_depth=2)

    # return outputs
    return outputs


@dijet_balance.step(
    name="calc_jer",
    inputs={
        "asym.{width,extrapolation}.{reco,gen}",
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
    # inflate inputs
    inputs = inflate_dict(inputs)

    # initialize storage for outputs
    outputs = {}

    # use reco level for main variable lookup
    variable_map = self.variable_map["reco"]

    # compute JER for main response
    response_key = self.calc_jer_main_response
    response_cfg = self.responses[response_key]

    # retrieve inputs for response
    r_inputs = inputs[response_key]
    r_outputs = outputs[response_key] = {}

    # check inputs for response
    check_hist_axes(
        hists=r_inputs,
        checks={
            "*.reco": set(
                set(response_cfg["all_vars"]["reco"]) -
                {
                    response_cfg["response_vars"]["reco"],
                    variable_map[self.extrapolation_var_key],
                },
            ),
            "*.gen": set(
                set(response_cfg["all_vars"]["gen"]) -
                {
                    response_cfg["response_vars"]["gen"],
                    variable_map[self.extrapolation_var_key],
                },
            ),
        },
    )

    # helper function for getting either extracted or
    # extrapolated widths
    def _get_width(level):
        # copy input histograms
        if self.calc_jer_use_extrapolation:
            # get extrapolated intercepts
            return r_inputs["extrapolation"][level]["intercepts"].copy(deep=False)
        else:
            # get extracted widths and use last (most inclusive) alpha bin
            h_inputs = r_inputs["width"][level].copy(deep=False)
            return h_inputs[{
                self.variable_map[level][self.extrapolation_var_key]: -1,
            }]

    # get input histograms
    h_widths = _get_width("reco")

    # if requested, subtract the gen-level results from the extrapolated widths
    if self.calc_jer_subtract_pli:
        h_widths_gen = _get_width("gen")
        h_widths = hist_sub_in_quadrature(h_widths, h_widths_gen)

        # add PLI-subtracted extrapolation result to output
        r_outputs["width_pli_subtracted"] = h_widths.copy()

    # get index on `category` axis corresponding to
    # the two computation methods
    categories = list(h_widths.axes["category"])
    index_methods = {
        m: categories.index(self.config_inst.get_category(m))
        for m in self.config_inst.x.method_categories
    }

    # calcuate JER for standard method
    jer_sm_val = h_widths[index_methods["sm"], :, :].values() * np.sqrt(2)
    jer_sm_err = np.sqrt(h_widths[index_methods["sm"], :, :].variances()) * np.sqrt(2)

    # get number of |eta| bins over which to average
    # when computing the reference JER using the SM
    n_bins_abseta = h_widths.axes[variable_map["abseta"]].index(self.calc_jer_max_abseta_sm_ref)

    # average over first few |eta| bins to get
    # reference JER for FE method
    jer_ref_val = np.mean(jer_sm_val[:n_bins_abseta, :], axis=0, keepdims=True)
    jer_ref_err = np.mean(jer_sm_err[:n_bins_abseta, :], axis=0, keepdims=True)

    # calculate JER for forward extension method
    jer_fe_val = np.sqrt(4 * h_widths[index_methods["fe"], :, :].values()**2 - jer_ref_val**2)
    term_probe = 4 * h_widths[index_methods["fe"], :, :].values() * h_widths[index_methods["fe"], :, :].variances()
    term_ref = jer_ref_val * jer_ref_err
    jer_fe_err = np.sqrt(term_probe**2 + term_ref**2) / jer_fe_val

    # create output histogram and view for filling
    h_jer = h_widths.copy()
    v_jer = h_jer.view()

    # write JER values to output histogram
    v_jer[index_methods["sm"], :, :].value = np.nan_to_num(jer_sm_val, nan=0.0)
    v_jer[index_methods["sm"], :, :].variance = np.nan_to_num(jer_sm_err**2, nan=0.0)
    v_jer[index_methods["fe"], :, :].value = np.nan_to_num(jer_fe_val, nan=0.0)
    v_jer[index_methods["fe"], :, :].variance = np.nan_to_num(jer_fe_err**2, nan=0.0)

    # add JER to output
    r_outputs["jer"] = h_jer.copy()

    # deflate outputs
    outputs = deflate_dict(outputs, max_depth=2)

    # return results
    return outputs


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
    # inflate inputs
    inputs = inflate_dict(inputs)

    # initialize storage for outputs
    outputs = {}

    # use reco level for main variable lookup
    variable_map = self.variable_map["reco"]

    # compute JER for main response
    response_key = self.calc_jer_main_response
    response_cfg = self.responses[response_key]

    # retrieve inputs for response
    r_inputs = inputs[response_key]
    r_outputs = outputs[response_key] = {}

    # check inputs for response
    check_hist_axes(
        hists=r_inputs,
        checks={
            "*": set(
                set(response_cfg["all_vars"]["reco"]) -
                {
                    response_cfg["response_vars"]["reco"],
                    variable_map[self.extrapolation_var_key],
                },
            ),
        },
    )

    # store views of inputs (copy to avoid modifying input)
    v_jers = {
        "data": r_inputs["jer"]["data"].view().copy(),
        "mc": r_inputs["jer"]["mc"].view().copy(),
    }

    # output histogram with scale factors
    h_sf = r_inputs["jer"]["data"].copy()
    v_sf = h_sf.view()

    v_sf.value = v_jers["data"].value / v_jers["mc"].value
    # inf values if mc is zero; add for precaution
    mask = np.fabs(v_sf.value) == np.inf  # account also for -inf
    v_sf.value[mask] = np.nan
    v_sf.value = np.nan_to_num(v_sf.value, nan=0.0)

    # Error propagation
    # x = data; y = mc; s_x = sigma x
    # x/y -> sqrt( ( s_x/y )**2 + ( (x*s_y)/y**2 )**2 )
    term1 = v_jers["data"].variance / v_jers["mc"].value
    term2 = (v_jers["data"].value * v_jers["mc"].variance) / v_jers["mc"].value**2
    v_sf.variance = np.sqrt(term1**2 + term2**2)
    # inf values if mc is zero; add for precaution
    v_sf.variance[mask] = np.nan  # account also for -inf
    v_sf.variance = np.nan_to_num(v_sf.variance, nan=0.0)

    # add JER to output
    r_outputs["jer_sf"] = h_sf.copy()

    # deflate outputs
    outputs = deflate_dict(outputs, max_depth=2)

    # return results
    return outputs


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
