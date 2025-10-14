# coding: utf-8

"""
Selector for synching with UHH2
"""

from operator import and_
from functools import reduce
from collections import defaultdict
from typing import Tuple

from columnflow.util import maybe_import

from columnflow.selection import Selector, SelectionResult, selector
from dijet.selection.default import default

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={
        default,
    },
    produces={
        default,
    },
    exposed=True,
    check_used_columns=True,
    check_produced_columns=True,
)
def uhh2(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:

    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # run default selector
    events, default_results = self[default](events, stats, **kwargs)
    results += default_results

    # UHH2-specific selection steps
    results += SelectionResult(
        steps={
            # ensure pTavg > 50 GeV
            "dijet_pt_avg_gt_50": ak.fill_none(events.dijets.pt_avg > 50, False),
        },
    )

    # results.event contains full selection mask. Sum over all steps.
    # Make sure all nans are present, otherwise next tasks fail
    results.event = reduce(and_, results.steps.values())
    results.event = ak.fill_none(results.event, False)

    return events, results
