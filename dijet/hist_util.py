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
from columnflow.types import TYPE_CHECKING, Any

np = maybe_import("numpy")
ak = maybe_import("awkward")
if TYPE_CHECKING:
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
