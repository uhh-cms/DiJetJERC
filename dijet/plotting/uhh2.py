# coding: utf-8
"""
Task for plotting UHH2 synchronization related things 
"""
from __future__ import annotations

import itertools
import law

from collections import OrderedDict
from functools import partial

from columnflow.util import maybe_import
from columnflow.tasks.plotting import PlotVariables2D
from columnflow.plotting.plot_util import (
    remove_residual_axis,
    apply_variable_settings,
    apply_process_settings,
    apply_process_scaling,
    apply_density,
    get_position,
    reduce_with,
)
from columnflow.plotting.plot_all import plot_all

from dijet.tasks.base import DiJetTask
from dijet.plotting.base import PlottingBaseTask, bin_skip_fn
from dijet.plotting.util import annotate_corner, get_bin_slug, get_bin_label, plot_xy

hist = maybe_import("hist")
np = maybe_import("numpy")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")

def plot_two_1d(
    hists: OrderedDict,
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable],
    shift_insts: list[od.Shift],
    style_config: dict | None = None,
    density: bool | None = False,
    shape_norm: bool | None = False,
    skip_legend: bool = False,
    cms_label: str = "wip",
    process_settings: dict | None = None,
    variable_settings: dict | None = None,
    **kwargs,
) -> plt.Figure:
    assert len(variable_insts) == 2, f"internal error: two variables expected, got {len(variable_insts)}"

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import mplhep

    # remove shift axis from histograms
    hists = remove_residual_axis(hists, "shift")

    hists, process_style_config = apply_process_settings(hists, process_settings)
    hists, variable_style_config = apply_variable_settings(hists, variable_insts, variable_settings)
    hists = apply_process_scaling(hists)
    if density:
        hists = apply_density(hists, density)

    # use CMS plotting style
    plt.style.use(mplhep.style.CMS)
    fig, ax = plt.subplots()

    # add all processes into 1 histogram
    h_sum = sum(list(hists.values())[1:], list(hists.values())[0].copy())

    # unit format on axes (could be configurable)
    unit_format = "{title} [{unit}]"

    # setup style config   
    default_style_config = {
        "ax_cfg": {
            "xlim": (
                min(variable_insts[0].x_min, variable_insts[1].x_min),
                max(variable_insts[0].x_max, variable_insts[1].x_max),
            ),
            "xlabel": "Variable value",
            "ylabel": "Entries",
            "xscale": "log" if (variable_insts[0].log_x or variable_insts[1].log_x) else "linear",
        },
        "rax_cfg": {
            "xlim": (
                min(variable_insts[0].x_min, variable_insts[1].x_min),
                max(variable_insts[0].x_max, variable_insts[1].x_max),
            ),
            "xlabel": "Variable value",
            "ylabel": "Ratio",
            "xscale": "log" if (variable_insts[0].log_x or variable_insts[1].log_x) else "linear",
        },
        "legend_cfg": {
            #"title": "Process" if len(hists.keys()) == 1 else "Processes",
            #"handles": [mpl.lines.Line2D([0], [0], lw=0) for proc_inst in hists.keys()],  # dummy handle
            #"labels": [proc_inst.label for proc_inst in hists.keys()],
            "ncol": 1,
            "loc": "upper right",
        },
        "cms_label_cfg": {
            "lumi": round(0.001 * config_inst.x.luminosity.get("nominal"), 1),  # /pb -> /fb
            "com": config_inst.campaign.ecm,
        },
        #"gridspec_cfg": {},
        "annotate_cfg": {
            "text": category_inst.label,
        },
    }
    style_config = law.util.merge_dicts(
        default_style_config,
        process_style_config,
        variable_style_config[variable_insts[0]],
        variable_style_config[variable_insts[1]],
        style_config,
        deep=True,
    )

    # apply style_config
    ax.set(**style_config["ax_cfg"])
    if not skip_legend:
        ax.legend(**style_config["legend_cfg"])

    if variable_insts[0].discrete_x:
        ax.set_xticks([], minor=True)
    if variable_insts[1].discrete_x:
        ax.set_yticks([], minor=True)

    # annotation of category label
    annotate_kwargs = {
        "text": "",
        "xy": (
            get_position(*ax.get_xlim(), factor=0.05, logscale=False),
            get_position(*ax.get_ylim(), factor=0.95, logscale=False),
        ),
        "xycoords": "data",
        "color": "black",
        "fontsize": 22,
        "horizontalalignment": "left",
        "verticalalignment": "top",
    }
    annotate_kwargs.update(default_style_config.get("annotate_cfg", {}))
    plt.annotate(**annotate_kwargs)

    # cms label
    if cms_label != "skip":
        label_options = {
            "wip": "Work in progress",
            "pre": "Preliminary",
            "pw": "Private work",
            "sim": "Simulation",
            "simwip": "Simulation work in progress",
            "simpre": "Simulation preliminary",
            "simpw": "Simulation private work",
            "od": "OpenData",
            "odwip": "OpenData work in progress",
            "odpw": "OpenData private work",
            "public": "",
        }
        cms_label_kwargs = {
            "ax": ax,
            "llabel": label_options.get(cms_label, cms_label),
            "fontsize": 22,
            "data": False,
        }

        cms_label_kwargs.update(style_config.get("cms_label_cfg", {}))
        mplhep.cms.label(**cms_label_kwargs)

    hists_1d = []
    for i_ax in range(2):
        h_ax = h_sum.project(i_ax)

        # normalize to 1.0 if requested
        if shape_norm:
            h_ax = h_ax / h_ax.sum().value

        # mask bins without any entries (variance == 0)
        h_view = h_ax.view()
        h_view.value[h_view.variance == 0] = np.nan

        hists_1d.append(h_ax)

    line_labels = [
        v.x_title
        for v in variable_insts
    ]
    line_colors = ["k", "r"]

    plot_config = {
        f"line_{i}": {
            "method": "draw_hist",
            "hist": hist_1d,
            "kwargs": {
                # "norm": line_norm,
                "label": line_labels[i],
                "color": line_colors[i],
                "error_type": "variance",
            },
            "ratio_kwargs": {
                  "norm": hists_1d[0].values(),  # use first variable as a reference
                  # FIXME: what if bins are different?
                  "color": line_colors[i],
            },
        }
        for i, hist_1d in enumerate(hists_1d)
    }

    return plot_all(plot_config, style_config, **kwargs)



class PlotTwoVariables1D(DiJetTask, PlotVariables2D):
    plot_function = "dijet.plotting.uhh2.plot_two_1d"
