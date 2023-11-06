# coding: utf-8

"""
Selection methods for HHtobbWW.
"""

from operator import and_
from functools import reduce
from collections import defaultdict
from typing import Tuple

from columnflow.util import maybe_import

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.cms.met_filters import met_filters
from columnflow.selection.cms.json_filter import json_filter

from columnflow.production.util import attach_coffea_behavior
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.processes import process_ids

from dijet.production.categories import category_ids
from dijet.production.weights import large_weights_killer
from dijet.production.dijet_balance import dijet_balance
from dijet.production.jet_assignment import jet_assignment
from dijet.selection.jet import jet_selection
from dijet.selection.dijet import dijet_selection
from dijet.selection.lepton_selection import lepton_selection
from dijet.selection.trigger import trigger_selection
from dijet.selection.cutflow_features import cutflow_features
from dijet.selection.stats import dijet_increment_stats

np = maybe_import("numpy")
ak = maybe_import("awkward")


def masked_sorted_indices(mask: ak.Array, sort_var: ak.Array, ascending: bool = False) -> ak.Array:
    """
    Helper function to obtain the correct indices of an object mask
    """
    indices = ak.argsort(sort_var, axis=-1, ascending=ascending)
    return indices[mask[indices]]


@selector(
    uses={
        met_filters, json_filter,
        category_ids, process_ids, attach_coffea_behavior,
        mc_weight, large_weights_killer,  # not opened per default but always required in Cutflow tasks
        jet_selection, lepton_selection, trigger_selection, dijet_selection,
        dijet_balance, jet_assignment, cutflow_features, dijet_increment_stats,
    },
    produces={
        met_filters, json_filter,
        category_ids, process_ids, attach_coffea_behavior,
        mc_weight, large_weights_killer,
        jet_selection, lepton_selection, trigger_selection, dijet_selection,
        dijet_balance, jet_assignment, cutflow_features, dijet_increment_stats,
    },
    exposed=True,
    check_used_columns=False,
    check_produced_columns=False,
)
def default(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:

    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)
        events = self[large_weights_killer](events, **kwargs)

    # ensure coffea behavior
    events = self[attach_coffea_behavior](events, **kwargs)

    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # MET filters
    events, met_filters_results = self[met_filters](events, **kwargs)
    results += met_filters_results

    # JSON filter (data-only)
    if self.dataset_inst.is_data:
        events, json_filter_results = self[json_filter](events, **kwargs)
        results += json_filter_results

    # # TODO Implement selection
    # # lepton selection
    events, results_lepton = self[lepton_selection](events, **kwargs)
    results += results_lepton

    # jet selection
    events, results_jet = self[jet_selection](events, **kwargs)
    results += results_jet

    # dijet balance for cutflow variables
    # TODO: Remove later
    events = self[jet_assignment](events, **kwargs)
    events = self[dijet_balance](events, **kwargs)

    # trigger selection
    # Uses pt_avg and the probe jet
    events, results_trigger = self[trigger_selection](events, **kwargs)
    results += results_trigger

    events, results_dijet = self[dijet_selection](events, **kwargs)
    results += results_dijet

    # create process ids
    events = self[process_ids](events, **kwargs)

    # build categories
    events = self[category_ids](events, results=results, **kwargs)

    # produce relevant columns
    events = self[cutflow_features](events, results.objects, **kwargs)

    # results.main.event contains full selection mask. Sum over all steps.
    # Make sure all nans are present, otherwise next tasks fail
    results.main["event"] = reduce(and_, results.steps.values())
    results.main["event"] = ak.fill_none(results.main["event"], False)

    self[dijet_increment_stats](events, results, stats, **kwargs)

    return events, results


# @default.init
# def default_init(self: Selector) -> None:
#     if self.config_inst.x("do_cutflow_features", False):
#         self.uses.add(cutflow_features)
#         self.produces.add(cutflow_features)

#     if not getattr(self, "dataset_inst", None) or self.dataset_inst.is_data:
#         return

#     self.uses.add(event_weights_to_normalize)
#     self.produces.add(event_weights_to_normalize)
