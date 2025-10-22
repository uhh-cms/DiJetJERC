# coding: utf-8

"""
Collection of helper functions for creating and handling histograms.
"""

from __future__ import annotations

__all__ = []

import functools
import law
import order as od

from columnflow.hist_util import (
    copy_axis,
    get_axis_kwargs,
    add_hist_axis,
    create_hist_from_variables,
    translate_hist_intcat_to_strcat,
    add_missing_shifts,
)
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
