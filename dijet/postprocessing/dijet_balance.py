# coding: utf-8

"""
Postprocessors for JER.
"""

from __future__ import annotations

import itertools as it

import law
import order as od

from columnflow.util import maybe_import
from columnflow.types import Sequence

from dijet.postprocessing import PostProcessor, postprocessor

from dijet.hist_util import hist_mean_variance, hist_fit_gaussian
from dijet.fit_util import CorrelatedFit

np = maybe_import("numpy")
ak = maybe_import("awkward")
hist = maybe_import("hist")


#
# utility functions
#

def check_inputs(inputs: dict[hist.Hist], axis_names: Sequence[str], level: str):
    """
    Check that all *inputs* contain the axes specified as *axis_names*. Raises
    a ``ValueError`` if one or more axes are missing.
    """
    # check that input histograms contain all required variables
    for input_name, input_hist in inputs.items():
        existing_axis_names = {a.name for a in input_hist.axes}
        missing_axes = set(axis_names) - existing_axis_names
        if missing_axes:
            missing_axes_str = ",".join(sorted(missing_axes))
            existing_axis_names_str = ",".join(sorted(existing_axis_names))
            raise ValueError(
                f"input histogram '{input_name}' for level '{level}' is missing axes; "
                f"expected: {missing_axes_str}, available: {existing_axis_names_str}",
            )


#
# create and intialize post-processor
#

dijet_balance = postprocessor(
    "dijet_balance",
    # user-defined parameters
    variable_map={
        # reco-level variables
        "reco": {
            # variable used for extrapolating raw response
            "alpha": "dijets_alpha",
            # variable used for binning
            "abseta": "probejet_abseta",
            "pt": "dijets_pt_avg",
            # variable representing the raw response
            "asymmetry": "dijets_asymmetry",  # TODO: rename to response
        },
        # gen-level variables
        "gen": {
            # variable used for extrapolating raw response
            "alpha": "dijets_alpha",  # TODO: change to alpha_gen?
            # variable used for binning
            "abseta": "probejet_abseta_gen",
            "pt": "dijets_pt_avg_gen",
            # variable representing the raw response
            "asymmetry": "dijets_asymmetry_gen",  # TODO: rename to response
        },
    },
    # keys of variables used for binning (not extrapolation or response itself)
    binning_var_keys=("abseta", "pt"),
    # rebinning factor to apply to asymmetry axis
    trim_tails_asymmetry_rebin_factor=2,
    # total fraction of response distribution to remove symetrically from both
    # ends in order to cut off non-Gaussian tails
    trim_tails_fraction=0.03,
    # method to use for extracting width
    extract_width_method="empirical",
    # whether to subtract the particle-level imbalance
    # (gen-level resolution) when calculating JER
    calc_jer_subtract_pli=True,
    # upper |eta| value until which to average
    # SM JER to use as a reference for the FE method
    calc_jer_max_abseta_sm_ref=1.131,
)


@dijet_balance.variables
def dijet_balance_variables(self: PostProcessor, task: law.Task, dataset_inst: od.Dataset):
    """
    Compute multidimensional variables used for filling input histograms.
    """
    variables = set()

    # determine valid levels
    levels = {"reco"}
    if dataset_inst.is_mc:
        levels.add("gen")

    # compute multi-dimensional input variables
    # from variable map and level information
    for level in levels:
        single_variables = self.variable_map[level].values()
        variables.add("-".join(single_variables))

    # return variables
    return variables


#
# register post-processing steps
#

@dijet_balance.step(
    name="trim_tails",
    inputs={"hist"},
    outputs={"asym", "asym_cut", "nevt", "cut_edges"},
    # TODO: separate out "trim_tails", "count_events" and "calc_quantiles" ?
    # TODO: rename 'asym' to 'response' (more general)
)
def trim_tails(
    self: PostProcessor,
    task: law.Task,
    inputs: dict[hist.Hist],
    level: str,
    **kwargs,
) -> dict[hist.Hist]:
    """
    Trim the (possibly non-Gaussian) tails of the response distributions.

    Inputs
    ------
    * ``hist``: a multi-dimensional histogram containing the raw response distributions

    Outputs
    -------
    * ``asym``: a multi-dimensional histogram containing the normalized raw response distributions
    * ``asym_cut``: a multi-dimensional histogram containing the normalized response distributions
                    after trimming non-Gaussian tails
    * ``nevt``: a multi-dimensional histogram containing the event yields (integral over response)
    * ``cut_edges``: a dict of histograms ``low`` and ``up`` containing the lower and upper bound
                     of the response interval after trimming non-Gaussian tails (for plotting)
    """
    # initialize storage for outputs
    outputs = {}

    # use correct level for variable lookup
    variable_map = self.variable_map[level]  # TODO: pass variable_map directly?

    # check that input hists contain the correct axes
    check_inputs(
        inputs=inputs,
        axis_names=variable_map.values(),
        level=level,
    )

    # copy input histogram
    h_in = inputs["hist"].copy(deep=False)

    # rebin asymmetry (TODO: use config binning?)
    h_in = h_in[{
        variable_map["asymmetry"]: hist.rebin(
            self.trim_tails_asymmetry_rebin_factor,
        ),
    }]

    # resolve axis names and indices
    axis_names = [a.name for a in h_in.axes]
    axis_indices = {
        var_key: axis_names.index(variable_map[var_key])
        for var_key in ("alpha", "asymmetry")
    }

    # work on view of histogram data
    v_in = h_in.view()

    # replace histogram contents with cumulative sum over
    # bins in the extrapolation variable (typically 'alpha')
    v_in.value = np.apply_along_axis(np.cumsum, axis=axis_indices["alpha"], arr=v_in.value)
    v_in.variance = np.apply_along_axis(np.cumsum, axis=axis_indices["alpha"], arr=v_in.variance)

    # get total event yield from integral over asymmetry distrivution
    # (note: we don't include over-/underflow bins here)
    integral = h_in.values().sum(axis=axis_indices["asymmetry"], keepdims=True)
    integral_var = h_in.variances().sum(axis=axis_indices["asymmetry"], keepdims=True)

    # store event yield as histogram in output
    h_nevts = h_in.copy()
    h_nevts = h_nevts[{variable_map["asymmetry"]: sum}]
    h_nevts.view().value = np.squeeze(integral)
    h_nevts.view().variance = np.squeeze(integral_var)
    outputs["nevt"] = h_nevts.copy(deep=False)

    # normalize histogram to integral over asymmetry
    v_in.value = v_in.value / integral
    v_in.variance = v_in.variance / integral**2

    # store full asymmetry distribution (including non-Gaussian tails)
    outputs["asym"] = h_in.copy(deep=False)

    #
    # next, trim non-Gaussian tails
    #

    # get cumulative response distribution
    fraction = np.cumsum(v_in.value, axis=axis_indices["asymmetry"])
    # TODO: last fraction should be 1 -> check

    # compute lower and upper quantiles for trimming
    quantile_lo = self.trim_tails_fraction / 2
    quantile_up = 1.0 - quantile_lo

    # find index of bins corresponding to quantiles
    ind_lo = np.apply_along_axis(np.searchsorted, axis_indices["asymmetry"], fraction, quantile_lo, side="left")
    ind_up = np.apply_along_axis(np.searchsorted, axis_indices["asymmetry"], fraction, quantile_up, side="right")

    # add back reduced dimension
    ind_lo = np.expand_dims(ind_lo, axis=axis_indices["asymmetry"])
    ind_up = np.expand_dims(ind_up, axis=axis_indices["asymmetry"])

    # obtain bin edges corresponding to requested quantiles
    asym_edges = h_in.axes[variable_map["asymmetry"]].edges
    asym_edges_lo = asym_edges[ind_lo]
    asym_edges_up = asym_edges[ind_up + 1]  # upper edge -> add 1

    # store bin edges in histogram format
    h_asym_edges_lo = h_in.copy()[{variable_map["asymmetry"]: sum}]
    h_asym_edges_up = h_asym_edges_lo.copy()
    h_asym_edges_lo.view().value = np.squeeze(asym_edges_lo)
    h_asym_edges_lo.view().variance[:] = 0.0
    h_asym_edges_up.view().value = np.squeeze(asym_edges_up)
    h_asym_edges_up.view().variance[:] = 0.0
    outputs["cut_edges"] = {
        "low": h_asym_edges_lo.copy(),
        "up": h_asym_edges_up.copy(),
    }

    # compute mask to select tail bins
    asym_centers = h_in.axes[variable_map["asymmetry"]].centers
    # asym_centers = asym_centers.reshape(1, 1, 1, 1, -1)  # TODO: not hard-coded
    asym_centers_shape = (1,) * (asym_edges_lo.ndim - 1) + (-1,)
    asym_centers = asym_centers.reshape(*asym_centers_shape)
    # asym_centers = np.expand_dims(asym_centers, axis=axis_indices["asymmetry"])
    mask = (asym_centers > asym_edges_lo) & (asym_centers < asym_edges_up)

    # set tail bins to zero for trimming
    v_in.value = np.where(mask, v_in.value, 0)
    v_in.variance = np.where(mask, v_in.variance, 0)

    # store asymmetry with trimmed non-Gaussian tails
    outputs["asym_cut"] = h_in.copy(deep=False)

    # return outputs
    return outputs


@dijet_balance.step(
    name="extract_width",
    inputs={"asym_cut", "nevt"},
    outputs={"width"},
)
def extract_width(
    self: PostProcessor,
    task: law.Task,
    inputs: dict[hist.Hist],
    level: str,
    **kwargs,
) -> dict[hist.Hist]:
    """
    Extract the width of the response distributions.

    Inputs
    ------
    * ``asym_cut``: a multi-dimensional histogram containing the response distributions
    * ``nevt``: a multi-dimensional histogram containing the event yields (integral over response)

    Outputs
    -------
    * ``width``: the width of the response distributions
    """
    # initialize storage for outputs
    outputs = {}

    # use correct level for variable lookup
    variable_map = self.variable_map[level]  # TODO: pass variable_map directly?

    # check that input hists contain the correct axes
    axis_names_no_asym = [
        var_name
        for var_key, var_name in variable_map.items()
        if var_key != "asymmetry"
    ]
    check_inputs(
        inputs={"asym_cut": inputs["asym_cut"]},
        axis_names=variable_map.values(),
        level=level,
    )
    check_inputs(
        inputs={"nevt": inputs["nevt"]},
        axis_names=axis_names_no_asym,
        level=level,
    )

    # copy input histogram
    h_asyms = inputs["asym_cut"].copy(deep=False)
    h_nevts = inputs["nevt"].copy(deep=False)

    # resolve axis names and indices
    axis_names = [a.name for a in h_asyms.axes]

    #
    # start main processing
    #

    # check that asymmetry axis is last (TODO: remove this limitation)
    if axis_names[-1] != variable_map["asymmetry"]:
        raise RuntimeError(
            "internal error: asymmetry axis must come last",
        )

    #
    # method choice: Gaussian fit or empirical standard deviation
    #

    if self.extract_width_method == "gaussian_fit":
        # fit Gaussian to input histograms along asymmetry axis
        h_fit_gaussian = hist_fit_gaussian(
            h_asyms,
            axis=variable_map["asymmetry"],
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
            h_asyms,
            axis=variable_map["asymmetry"],
        )

        # calculate standard deviation and its errors
        # note: the error on standard deviation is analogous to the
        # implementation in ROOT::TH1:
        # https://root.cern/doc/v630/TH1_8cxx_source.html#l07520
        h_stds_emp = h_mean_variance[..., "variance"]
        v_stds_emp = h_stds_emp.view()
        with np.errstate(invalid="ignore"):
            v_stds_emp.variance = v_stds_emp.value / (2 * h_nevts.values())
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
    outputs["width"] = h_stds

    # return outputs
    return outputs


@dijet_balance.step(
    name="extrapolate_width",
    inputs={"width", "nevt"},
    outputs={"extrapolation"},
)
def extrapolate_width(
    self: PostProcessor,
    task: law.Task,
    inputs: dict[hist.Hist],
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
    # initialize storage for outputs
    outputs = {}

    # use correct level for variable lookup
    variable_map = self.variable_map[level]  # TODO: pass variable_map directly?

    # check that input hists contain the correct axes
    axis_names_no_asym = [
        var_name
        for var_key, var_name in variable_map.items()
        if var_key != "asymmetry"
    ]
    check_inputs(
        inputs=inputs,
        axis_names=axis_names_no_asym,
        level=level,
    )

    # copy input histograms
    h_stds = inputs["width"].copy(deep=False)
    h_nevts = inputs["nevt"].copy(deep=False)

    # Get max alpha for fit; usually 0.3
    amax = 0.3  # TODO: define in config
    h_stds = h_stds[{variable_map["alpha"]: slice(0, hist.loc(amax))}]
    h_nevts = h_nevts[{variable_map["alpha"]: slice(0, hist.loc(amax))}]
    # exclude 0, the first bin, from alpha edges
    alphas = h_stds.axes[variable_map["alpha"]].edges[1:]

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
        tmp_evts = h_nevts[h_slice]
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
    h_intercepts = h_intercepts[{variable_map["alpha"]: sum}]
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
    outputs["extrapolation"] = {
        "intercepts": h_intercepts,
        "slopes": h_slopes,
    }

    # return outputs
    return outputs


@dijet_balance.step(
    name="calc_jer",
    inputs={"extrapolation_reco", "extrapolation_gen"},
    outputs={"jer"},
)
def calc_jer(
    self: PostProcessor,
    task: law.Task,
    inputs: dict[hist.Hist],
    **kwargs,
) -> dict[hist.Hist]:
    """
    Subtract the particle-level imbalance (i.e. the extrapolated gen-level response width)
    from the reconstructed-level response width.

    Inputs
    ------
    * ``extrapolation_reco``: a multi-dimensional histogram containing the extrapolated widths on reco-level
    * ``extrapolation_gen``: a multi-dimensional histogram containing the extrapolated widths on gen-level

    Outputs
    -------
    * ``jer``: the JER, calculated from the extrapolated widths (TODO: document formulae)
    """
    # initialize storage for outputs
    outputs = {}

    # use reco level for main variable lookup
    variable_map = self.variable_map["reco"]

    # check that input hists contain the correct axes
    for level in ("reco", "gen"):
        variable_map_for_check = self.variable_map[level]
        axis_names_no_asym_no_extp = [
            var_name
            for var_key, var_name in variable_map_for_check.items()
            if var_key not in ("asymmetry", "alpha")
        ]
        input_key = f"extrapolation_{level}"
        check_inputs(
            inputs={input_key: inputs[input_key]},
            axis_names=axis_names_no_asym_no_extp,
            level=level,
        )

    # copy input histograms
    h_widths = inputs["extrapolation_reco"].copy()

    # if requested, subtract the gen-level results from the extrapolated widths
    if self.calc_jer_subtract_pli:
        h_widths_gen = inputs["extrapolation_gen"].copy()
        values = np.sqrt(np.maximum(
            h_widths.values()**2 - h_widths_gen.values()**2,
            0.0,
        ))
        # Gaussian error propagation
        variances = np.nan_to_num(
            (
                h_widths.variances() * h_widths.values()**2 +
                h_widths_gen.variances() * h_widths_gen.values()**2
            ) / values**2,
            nan=0.0,
        )

        # save subtracted values back in h_widths
        v_widths = h_widths.view()
        v_widths.value = values
        v_widths.variance = variances

        # add PLI-subtracted extrapolation result to output
        outputs["extrapolation_pli_subtracted"] = h_widths.copy()

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
    outputs["jer"] = h_jer.copy()

    # return results
    return outputs


@dijet_balance.step(
    name="calc_sf",
    inputs={"jer_data", "jer_mc"},
    outputs={"sf"},
)
def calc_sf(
    self: PostProcessor,
    task: law.Task,
    inputs: dict[hist.Hist],
    **kwargs,
) -> dict[hist.Hist]:
    """
    Calculate the jet energy resolution scale factor from the JER in data and MC.

    Inputs
    ------
    * ``jer_data``: a multi-dimensional histogram containing the JER in data
    * ``jer_mc``: a multi-dimensional histogram containing the JER in MC

    Outputs
    -------
    * ``sf``: the JER scale factor, calculated as the ratio of JERs in data and MC
    """
    variable_map = self.variable_map["reco"]

    axis_names_no_asym_no_extp = [
        var_name
        for var_key, var_name in variable_map.items()
        if var_key not in ("asymmetry", "alpha")
    ]
    check_inputs(
        inputs=inputs,
        axis_names=axis_names_no_asym_no_extp,
        level="reco",
    )

    # store views of inputs (copy to avoid modifying input)
    v_jers = {
        "data": inputs["jer_data"].view().copy(),
        "mc": inputs["jer_mc"].view().copy(),
    }

    # output histogram with scale factors
    h_sf = inputs["jer_data"].copy()
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

    return {
        "sf": h_sf,
    }


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
