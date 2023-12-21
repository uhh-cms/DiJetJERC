# coding: utf-8

"""
Stat-related methods.
"""

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.stats import increment_stats
from columnflow.production.cms.btag import btag_weights
from dijet.production.weights import event_weights_to_normalize
from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={increment_stats, btag_weights, event_weights_to_normalize},
)
def dijet_increment_stats(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    stats: dict,
    **kwargs,
) -> ak.Array:
    # collect important information from the results
    event_mask = results.event

    # weight map definition
    weight_map = {
        # "num" operations
        "num_events": Ellipsis,  # all events
        "num_events_selected": event_mask,  # selected events only
        # "sum" operations
    }

    if self.dataset_inst.is_mc:
        weight_map["sum_mc_weight"] = events.mc_weight  # weights of all events
        weight_map["sum_mc_weight_selected"] = (events.mc_weight, event_mask)  # weights of selected events

    # Build weight_map to weight_map_per_process
    group_map = {
        "process": {
            "values": events.process_id,
            "mask_fn": (lambda v: events.process_id == v),
        },
    }

    # group_combinations = [("process")]

    self[increment_stats](
        events,
        results,
        stats,
        weight_map=weight_map,
        group_map=group_map,
        # group_combinations=group_combinations,
        **kwargs,
    )

    return events
