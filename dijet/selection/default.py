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
from columnflow.production.categories import category_ids
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.processes import process_ids

from dijet.production.weights import large_weights_killer
from dijet.production.jet_assignment import jet_assignment
from dijet.selection.jet import jet_selection
from dijet.selection.dijet import dijet_selection
from dijet.selection.lepton import lepton_selection
from dijet.selection.trigger import trigger_selection
from dijet.selection.cutflow_features import cutflow_features
from dijet.selection.stats import dijet_increment_stats

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={
        met_filters, json_filter,
        category_ids, process_ids, attach_coffea_behavior,
        mc_weight, large_weights_killer,
        jet_selection, lepton_selection, trigger_selection, dijet_selection,
        jet_assignment, cutflow_features, dijet_increment_stats,
    },
    produces={
        met_filters, json_filter,
        category_ids, process_ids, attach_coffea_behavior,
        mc_weight, large_weights_killer,
        jet_selection, lepton_selection, trigger_selection, dijet_selection,
        jet_assignment, cutflow_features, dijet_increment_stats,
    },
    exposed=True,
    check_used_columns=True,
    check_produced_columns=True,
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

    # ensure coffea behavior is attached to collections
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

    # lepton selection
    events, results_lepton = self[lepton_selection](events, **kwargs)
    results += results_lepton

    # jet selection
    events, results_jet = self[jet_selection](events, **kwargs)
    results += results_jet

    # retrieve jet mask to use in subsequent selection step
    jet_mask = results_jet.objects.Jet.Jet

    # trigger selection
    # Uses pt_avg and the probe jet
    if self.dataset_inst.is_data:
        events, results_trigger = self[trigger_selection](events, jet_mask=jet_mask, **kwargs)
        results += results_trigger

    # dijet selection
    events, results_dijet = self[dijet_selection](events, jet_mask=jet_mask, **kwargs)
    results += results_dijet

    # ad-hoc pTavg/HT cut in MC to remove unphysical events
    if self.dataset_inst.x("ht_range", None) is not None:
        pt_avg = (events.probe_jet.pt + events.reference_jet.pt) / 2
        ad_hoc_mask = (pt_avg < 1.5 * self.dataset_inst.x.ht_range[0])
        ad_hoc_sel = ak.fill_none(ad_hoc_mask, True)
        results.steps["ad_hoc_ptavg_ht_cut"] = ad_hoc_sel

    # create process ids
    events = self[process_ids](events, **kwargs)

    # build categories
    events = self[category_ids](events, results=results, **kwargs)

    # produce relevant columns
    events = self[cutflow_features](events, results.objects, **kwargs)

    # results.event contains full selection mask. Sum over all steps.
    # Make sure all nans are present, otherwise next tasks fail
    results.event = reduce(and_, results.steps.values())
    results.event = ak.fill_none(results.event, False)

    self[dijet_increment_stats](events, results, stats, **kwargs)

    return events, results


# @default.init
# def default_init(self: Selector) -> None:
#     if self.config_inst.x("do_cutflow_features", False):
#         self.uses.add(cutflow_features)
#         self.produces.add(cutflow_features)

#     self.uses.add(event_weights_to_normalize)
#     self.produces.add(event_weights_to_normalize)
