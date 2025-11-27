# coding: utf-8

"""
Collection of helper functions for creating and handling histograms.
"""

from __future__ import annotations

__all__ = []

import law

from columnflow.hist_util import add_hist_axis
from columnflow.columnar_util import flat_np_view
from columnflow.util import maybe_import
from columnflow.types import Any


np = maybe_import("numpy")
ak = maybe_import("awkward")
hist = maybe_import("hist")

logger = law.logger.get_logger(__name__)


def fill_hist(
    h: hist.Hist,
    data: ak.Array | np.array | dict[str, ak.Array | np.array],
    *,
    last_edge_inclusive: bool | None = None,
    fill_kwargs: dict[str, Any] | None = None,
) -> None:
    """
    Fills a histogram *h* with data from an awkward array, numpy array or nested dictionary *data*. The data is assumed
    to be structured in the same way as the histogram axes. If *last_edge_inclusive* is *True*, values that would land
    exactly on the upper-most bin edge of an axis are shifted into the last bin. If it is *None*, the behavior is
    determined automatically and depends on the variable axis type. In this case, shifting is applied to all continuous,
    non-circular axes.
    """
    import hist
    import boost_histogram as bh

    if fill_kwargs is None:
        fill_kwargs = {}

    # helper to decide whether the variable axis qualifies for shifting the last bin
    def allows_shift(ax) -> bool:
        return ax.traits.continuous and not ax.traits.circular

    # determine the axis names, figure out which which axes the last bin correction should be done
    axis_names = []
    correct_last_bin_axes = []
    for ax in h.axes:
        axis_names.append(ax.name)
        # include values hitting last edge?
        if not len(ax.widths) or not isinstance(ax, hist.axis.Variable):
            continue
        if (last_edge_inclusive is None and allows_shift(ax)) or last_edge_inclusive:
            correct_last_bin_axes.append(ax)

    # check data
    if not isinstance(data, dict):
        if len(axis_names) != 1:
            raise ValueError("got multi-dimensional hist but only one-dimensional data")
        data = {axis_names[0]: data}
    else:
        for name in axis_names:
            if name not in data and name not in fill_kwargs:
                raise ValueError(f"missing data for histogram axis '{name}'")

    # correct last bin values
    for ax in correct_last_bin_axes:
        right_egde_mask = ak.flatten(data[ax.name], axis=None) == ax.edges[-1]
        if np.any(right_egde_mask):
            data[ax.name] = ak.copy(data[ax.name])
            flat_np_view(data[ax.name])[right_egde_mask] -= ax.widths[-1] * 1e-5

    # check if conversion to records is needed
    arr_types = (ak.Array, np.ndarray)
    vals = list(data.values())
    convert = (
        # values is a mixture of singular and array types
        (any(isinstance(v, arr_types) for v in vals) and not all(isinstance(v, arr_types) for v in vals)) or
        # values contain at least one array with more than one dimension
        any(isinstance(v, arr_types) and v.ndim != 1 for v in vals)
    )

    # actual conversion
    if convert:
        arrays = ak.flatten(ak.cartesian(data))
        data = {field: arrays[field] for field in arrays.fields}
        del arrays

    # add 'sample' kwarg containing values of variable to be
    # profiled
    if h.storage_type == bh.storage.WeightedMean:
        profile_variable = h.axes[-1].name
        assert profile_variable in data, "internal error: variable to be profiles not found in fill dict"
        fill_kwargs["sample"] = data[profile_variable]

    # fill
    h.fill(**fill_kwargs, **data)


def create_hist_from_variables(
    *variable_insts,
    categorical_axes: tuple[tuple[str, str]] | None = None,
    weight: bool = True,
    storage: str | None = None,
) -> hist.Hist:
    import hist

    histogram = hist.Hist.new

    # additional category axes
    if categorical_axes:
        for name, axis_type in categorical_axes:
            if axis_type in ("intcategory", "intcat"):
                histogram = histogram.IntCat([], name=name, growth=True)
            elif axis_type in ("strcategory", "strcat"):
                histogram = histogram.StrCat([], name=name, growth=True)
            else:
                raise ValueError(f"unknown axis type '{axis_type}' in argument 'categorical_axes'")

    # requested axes from variables
    for variable_inst in variable_insts:
        histogram = add_hist_axis(histogram, variable_inst)

    # add the storage
    if storage is None:
        # use weight value for backwards compatibility
        storage = "weight" if weight else "double"
    else:
        storage = storage.lower()
    if storage == "weight":
        histogram = histogram.Weight()
    elif storage == "weighted_mean":
        histogram = histogram.WeightedMean()
    elif storage == "double":
        histogram = histogram.Double()
    else:
        raise ValueError(f"unknown storage type '{storage}'")

    return histogram


#
# functions for working with histogram data
#

def hist_apply_along_axis(
    h: hist.Hist,
    func: callable,
    axis: str | int,
    storage_cls: hist.storage.Storage | None = None,
    new_axes: list[hist.axis.AxesMixin] | None = None,
) -> hist.Hist | hist.storage.Storage:
    """
    Apply a function to histogram contents in slices along the given axis.

    The user-supplied function `func(struct_arr, **kwargs)` takes a NumPy
    structured array `struct_arr` as an input, whose fields correspond to the
    values tracked by the histogram storage. For example, for a histogram using
    the `Weight` storage type, the array of values and variances can be accessed
    via `struct_arr.value` and `struct_arr.variance` inside the function.

    The following `**kwargs` are passed to `func`:
        - `ax` (the axis the function is being applied to)
        - `dtype` (NumPy structured array dtype expected by the output
                   histogram storage)

    The function `func` should return a value that can be assigned to a
    structured array containing the fields tracked by the storage type.
    The simplest return type is a tuple, in which case the tuple elements
    are taken to correspond to the storage fields, in the order defined by
    the storage. For example, for `Weight` storage, the first element should
    contain the result used for the `value` field, and the second for `variance`.

    Optionally, a different storage class for the output histogram can be provided
    via `storage_cls`.

    *Advanced use*: `func` can also return an array with extra dimensions, in
    which case additional axes will be added to the histogram. By default, these
    will be `IntCategory` axes indexing the values contained in the return array
    starting with zero. Optionally, a list of Axis objects `new_axes` can be passed
    to this function and will be used when creating the output histogram. The
    number and length of Axis objects has to match the shape of the array
    returned by `func`. s

    A histogram with the same axis structure as `h` is returned, except
    for the `axis`, which is dropped and replaced by a storage element
    whose values are set by the array returned by `func`.

    Parameters
    ----------

    h: hist.Hist
        input histogram

    func: callable
        function to apply to the histogram contents

    axis: str | int
        axis to which the function should be applied

    storage_cls: type(hist.storage.Storage) (optional)
        class to use as a storage for the output histogram

    Return
    ------

    h_out: hist.Hist | hist.Storage
        output histogram (or Storage object if no axes left)
    """
    import hist
    import boost_histogram as bh

    # check storage type
    if h.storage_type != bh.storage.Weight:
        raise ValueError(
            f"unsupported storage type '{h.storage_type}'; "
            f"expected '{bh.storage.Weight}'",
        )

    # get axis index
    if isinstance(axis, str):
        axis_names = [a.name for a in h.axes]
        if axis not in axis_names:
            raise ValueError(f"unknown axis: {axis}")
        axis_index = axis_names.index(axis)

    elif isinstance(axis, int):
        if axis >= len(h.axes) or axis < -len(h.axes):
            raise IndexError(f"axis index {axis} out of range")
        axis_index = axis + len(h.axes) if axis < 0 else axis

    else:
        raise TypeError(
            f"expected 'str' or 'int' type for axis specification, got {type(axis)}",
        )

    # determine basic output axes (all but one input)
    if storage_cls is None:
        storage_cls = h.storage_type
    axes_out = [ax for i, ax in enumerate(h.axes) if i != axis_index]
    shape_out = tuple(len(a) for a in axes_out)

    # apply function to histogram contents
    f_out = np.apply_along_axis(
        func1d=func,
        axis=axis_index,
        arr=h.view(),
        # kwargs passed to `func` follow
        ax=h.axes[axis_index],
        dtype=hist.Hist(storage=storage_cls).view().dtype,
    )

    # add axes as needed to accommodate potential extra dimensions
    # of array returned by `func`
    shape_out_add = f_out.shape[len(shape_out):]  # dimensions added by `func`
    if new_axes is None:
        new_axes = [
            hist.axis.IntCategory(range(add_dim_len))
            for add_dim_len in shape_out_add
        ]

    # check number of axes
    if len(new_axes) != len(shape_out_add):
        raise ValueError(
            f"number of axes supplied as `new_axes` ({len(new_axes)}) "
            "does not match the number of additional dimensions inserted "
            f"({len(shape_out_add)})",
        )

    # check length of axes
    for new_axis, add_dim_len in zip(new_axes, shape_out_add):
        if add_dim_len != len(new_axis):
            raise ValueError(
                f"length of axis does not match the number of additional "
                f"dimensions inserted ({add_dim_len}): ({new_axis})",
            )

    # add the axes to the histogram
    axes_out.extend(new_axes)

    # create output histogram
    h_out = hist.Hist(*axes_out, storage=storage_cls)

    # write results to histogram storage
    h_out[...] = f_out

    # return output hist (or naked storage if no axes left)
    return h_out if axes_out else h_out[()]


#
# functions for operations on histogram contents
#

# TODO: rewrite using classes

def _mean_variance_impl(edges, values, variances):
    """
    Compute the mean and variance of a histogram
    """
    edges = np.asarray(edges)
    centers = 0.5 * (edges[1:] + edges[:-1])

    # sum of bin values and variances
    sum_values = np.sum(values)
    sum_variances = np.sum(variances)

    # compute average from bin centers and values
    num_avg = np.tensordot(values, centers, ([0], [0]))
    with np.errstate(invalid="ignore"):
        h_avg = num_avg / sum_values

    # compute variance from distance to average
    num_var = np.tensordot(values, (centers - h_avg)**2, ([0], [0]))
    with np.errstate(invalid="ignore"):
        h_var = num_var / sum_values

    # compute standard error on mean
    num_var = np.tensordot(variances, centers**2, ([0], [0]))
    with np.errstate(invalid="ignore"):
        h_avg_err = num_var / (sum_values ** 2) - sum_variances * (num_avg / sum_values ** 2) ** 2

    # compute standard error on variance
    h_var_err = np.zeros_like(h_var)
    with np.errstate(invalid="ignore"):
        h_var_err = h_var / (2 * sum_values)

    return (h_avg, h_avg_err), (h_var, h_var_err)


def _mean_variance_func1d(hist_data, ax, dtype):
    """
    Compute the mean and variance of the histogram data along the specified axis.
    """
    # pass the bin edges and contents to the mean/variance implementation
    (h_avg, h_avg_err), (h_var, h_var_err) = _mean_variance_impl(
        ax.edges,
        hist_data.value,
        hist_data.variance,
    )

    # Gaussian parameter values and variances
    avg_arr = np.rec.fromarrays(
        np.array([h_avg, h_avg_err]),
        dtype=dtype,
    )
    var_arr = np.rec.fromarrays(
        np.array([h_var, h_var_err]),
        dtype=dtype,
    )

    # return the values of interest
    return np.stack([avg_arr, var_arr], axis=-1)


def _fit_gaussian_impl(edges, values, variances):
    """
    Perform a maximum likelihood fit on a histogram
    using the BinnedNLL cost function from IMinuit.
    """
    from iminuit import Minuit
    from iminuit.cost import BinnedNLL
    from scipy.stats import norm

    # compute the cumulative distribution function (CDF)
    # of the probability sumsity that should be fit
    def cdf(xe, mu, sigma):
        return norm(loc=mu, scale=sigma).cdf(xe)

    # set up a cost function using the defined CDF
    cost_func = BinnedNLL(
        n=np.vstack([values, variances]).T,
        xe=edges,
        cdf=cdf,
    )

    # calculate empirical mean and variance of histogram
    (h_avg, _), (h_var, _) = _mean_variance_impl(edges, values, variances)

    # initialize and run a NLL fit using Minuit
    m = Minuit(cost_func, mu=h_avg, sigma=np.sqrt(h_var))
    m.migrad()

    # return the fit result object
    return m


def _fit_gaussian_func1d(hist_data, ax, dtype):
    """
    Fit a Gaussian probability density to the histogram data along the specified axis.
    """
    # pass the bin edges and contents to the fit implementation
    fit_result = _fit_gaussian_impl(
        ax.edges,
        hist_data.value,
        hist_data.variance,
    )

    mu = fit_result.params[0].value
    mu_var = fit_result.params[0].error ** 2
    sigma = fit_result.params[1].value
    sigma_var = fit_result.params[1].error ** 2

    # Gaussian parameter values and variances
    mu_arr = np.rec.fromarrays(
        np.array([mu, mu_var]),
        dtype=dtype,
    )
    sigma_arr = np.rec.fromarrays(
        np.array([sigma, sigma_var]),
        dtype=dtype,
    )

    # cost function value at minimum
    fval_arr = np.rec.fromarrays(
        np.array([fit_result.fval, np.zeros_like(fit_result.fval)]),
        dtype=dtype,
    )

    # number of degrees of freedom
    ndof_arr = np.rec.fromarrays(
        np.array([fit_result.ndof, np.zeros_like(fit_result.ndof)]),
        dtype=dtype,
    )

    # return the values of interest
    return np.stack([mu_arr, sigma_arr, fval_arr, ndof_arr], axis=-1)


def hist_fit_gaussian(h: hist.Histogram, axis: str | int = -1) -> hist.Histogram:
    """
    Fit a Gaussian probability density to a histogram *h* along a specified *axis*.
    If no *axis* is given, the last axis will be used by default.

    Returns a histogram with *axis* removed and replaced by an `StrCategory`
    axis containing the fit results. The axis contains the following entries:
    - "mu": mean of the fitted Gaussian
    - "sigma": standard deviation of the fitted Gaussian
    - "fval": value of the cost function at the minimum
    - "ndof": number of degrees of freedom of the fit
    """
    return hist_apply_along_axis(
        h, _fit_gaussian_func1d, axis,
        new_axes=[
            hist.axis.StrCategory(
                ["mu", "sigma", "fval", "ndof"],
                label="Fit result",
                name="fit",
            ),
        ],
    )


def hist_mean_variance(h: hist.Histogram, axis: str | int = -1) -> hist.Histogram:
    """
    Compute the mean and variance of a histogram *h* along a specified *axis*.
    If no *axis* is given, the last axis will be used by default.

    Returns a histogram with *axis* removed and replaced by an `StrCategory`
    axis containing the empirical statistics. The axis contains the following entries:
    - "mean": mean of the histogram along the specified axis
    - "variance": variance of the histogram along the specified axis
    """
    return hist_apply_along_axis(
        h, _mean_variance_func1d, axis,
        new_axes=[
            hist.axis.StrCategory(
                ["mean", "variance"],
                label="Empirical stats",
                name="stats",
            ),
        ],
    )


def _hist_binop_in_quadrature(
    op_name: str,
    hist_a: hist.Histogram,
    hist_b: hist.Histogram,
):
    """
    Given two identically-binned histograms, return
    a histogram containing their difference in quadrature,
    taking error propagation into account.
    """
    # fail if shapes are incompatible
    if hist_a.view().shape != hist_b.view().shape:
        msg = f"{hist_a.view().shape!r}, {hist_b.view().shape!r}"
        raise ValueError(
            f"histograms have incompatible storage shapes: {msg}",
        )

    # retrieve binary operator function
    import operator as op
    if op_name not in ("add", "sub") or (op_func := getattr(op, op_name, None)) is None:
        raise ValueError(
            f"invalid or unsupported binary operator: {op_name}",
        )

    # operate on copies
    h_a = hist_a.copy()
    h_b = hist_b.copy()

    # obtain output values by subtracting in quadrature
    # (replacing imaginary values with zero)
    values = np.sqrt(np.maximum(
        op_func(h_a.values()**2, h_b.values()**2),
        0.0,
    ))

    # obtain variances from Gaussian error propagation
    variances = np.nan_to_num(
        (
            h_a.variances() * h_a.values()**2 +
            h_b.variances() * h_b.values()**2
        ) / values**2,
        nan=0.0,
    )

    # save subtracted values back in h_a
    view = h_a.view()
    view.value = values
    view.variance = variances

    # return
    return h_a


def hist_add_in_quadrature(
    hist_a: hist.Histogram,
    hist_b: hist.Histogram,
):
    """
    Given two identically-binned histograms, return
    a histogram containing their sum in quadrature.
    Variances are obtained via Gaussian error propagation.
    """
    return _hist_binop_in_quadrature("add", hist_a, hist_b)


def hist_sub_in_quadrature(
    hist_a: hist.Histogram,
    hist_b: hist.Histogram,
):
    """
    Given two identically-binned histograms, return
    a histogram containing their difference in quadrature.
    Variances are obtained via Gaussian error propagation.
    """
    return _hist_binop_in_quadrature("sub", hist_a, hist_b)
