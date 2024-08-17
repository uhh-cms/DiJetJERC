# coding: utf-8
from __future__ import annotations

import order as od


def dot_to_p(var):
    return f"{var}".replace(".", "p")


def get_bin_label(var_inst: od.Variable, bin_edges: tuple[float]):
    """
    Get a representation of a bin using function stored under
    'bin_label' in variable aux data.
    """
    def default_bin_label_func(self, bin_edges):
        maybe_unit = (
            f" {self.unit}"
            if self.unit and self.unit not in ("1", 1)
            else ""
        )
        return " < ".join([
            self.x("bin_label_format", lambda x: f"{x:g}")(bin_edges[0]),
            self.x_title,
            self.x("bin_label_format", lambda x: f"{x:g}")(bin_edges[1]),
        ]) + maybe_unit

    func = var_inst.x("bin_label_func", default_bin_label_func)
    return func(var_inst, bin_edges)


def get_bin_slug(var_inst: od.Variable, bin_edges: tuple[float]):
    """
    Get a plain-text representation of a bin using function stored under
    'bin_slug' in variable aux data.
    """
    def default_bin_slug_func(self, bin_edges):
        return "_".join([
            self.x.slug_name,
            self.x("bin_slug_format", dot_to_p)(bin_edges[0]),
            self.x("bin_slug_format", dot_to_p)(bin_edges[1]),
        ])

    func = var_inst.x("bin_slug_func", default_bin_slug_func)
    return func(var_inst, bin_edges)


def annotate_corner(ax, text, loc="upper left", xy_offset=None, ha=None, va=None):
    """
    Add text in a corner of the Axes *ax*. Use *xy_offset* to specify the
    horizontal and vertical distance to the plot margin (in points).
    """
    # parse location
    loc_y, loc_x = loc.split(maxsplit=1)

    # resolve default alignments
    ha = ha or {"left": "left", "right": "right"}[loc_x]
    va = va or {"lower": "bottom", "upper": "top"}[loc_y]

    # resolve default offset
    if xy_offset:
        offset_x, offset_y = xy_offset
    else:
        default_offset = 20
        offset_x = {"left": 1, "right": -1}[loc_x] * default_offset
        offset_y = {"lower": 1, "upper": -1}[loc_y] * default_offset

    # resolve xy text coordinates
    loc_x = {"left": 0, "right": 1}[loc_x]
    loc_y = {"lower": 0, "upper": 1}[loc_y]

    return ax.annotate(
        text,
        xy=(loc_x, loc_y),
        xycoords="axes fraction",
        xytext=(offset_x, offset_y),
        textcoords="offset points",
        horizontalalignment=ha,
        verticalalignment=va,
    )


def plot_xy(x, y, xerr=None, yerr=None, method=None, ax=None, **kwargs):
    """
    Base function for drawing a series of xy values, possibly including
    uncertainties. *method* needs to be a valid method implemented by
    matplotlib *Axes* objects. *kwargs* can be supplied to the method.
    In general these  need to be supported by the method, but some
    adapting of known *kwargs* is done.
    """
    import matplotlib.pyplot as plt

    # set figure and axes, creating them if no `ax` is supplied
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.gcf(), plt.gca()

    # handle argument default values
    method = method or "errorbar"

    # resolve method function
    method_func = getattr(ax, method, None)
    if method_func is None:
        raise ValueError(f"invalid plot method '{method}'")

    # adapt kwargs to work with method
    if method == "bar":
        kwargs.update(
            align="center",
            width=2 * xerr,
            yerr=yerr,
        )
    elif method == "step":
        kwargs.pop("xerr", None)
        kwargs.pop("yerr", None)
        kwargs.pop("edgecolor", None)
    else:
        kwargs["xerr"] = xerr
        kwargs["yerr"] = yerr

    # call method
    artist = method_func(
        x.flatten(),
        y.flatten(),
        **kwargs,
    )

    # return figure, axes and artist
    return fig, ax, artist
