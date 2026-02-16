# coding: utf-8

"""
Histogram hooks.
"""

from __future__ import annotations

from functools import partial

import law
import order as od

from columnflow.util import maybe_import  # , DotDict
from columnflow.types import TYPE_CHECKING

np = maybe_import("numpy")
if TYPE_CHECKING:
    hist = maybe_import("hist")

logger = law.logger.get_logger(__name__)


def cumsum(
    task,
    hists: hist.Histogram,
    reverse: bool = False,
    **kwargs,
):
    for config_inst, proc_hists in hists.items():
        for proc_inst, proc_hist in proc_hists.items():
            if reverse:
                proc_hist.values()[...] = np.cumsum(proc_hist.values()[..., ::-1], axis=-1)[..., ::-1]
            else:
                proc_hist.values()[...] = np.cumsum(proc_hist.values(), axis=-1)

    return hists


def hist_check(
    task,
    hists: hist.Histogram,
    **kwargs,
):
    from IPython import embed
    embed()

    return hists


def add_hist_hooks(analysis_inst: od.Analysis) -> None:
    """
    Add histogram hooks to an analysis.
    """
    # add hist hooks to analysis instance
    analysis_inst.x.hist_hooks = {
        "cumsum": cumsum,
        "cumsum_reverse": partial(cumsum, reverse=True),
        "hist_check": hist_check,
    }
