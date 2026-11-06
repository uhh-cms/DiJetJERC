# coding: utf-8

"""
Producer to claclulate the MC truth pt response (revo pt / gen pt)
"""

from __future__ import annotations

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT
from columnflow.production import Producer, producer
from columnflow.production.util import attach_coffea_behavior, delta_r_match, delta_r_match_multiple


np = maybe_import("numpy")
ak = maybe_import("awkward")


custom_collections = {
    "Jet": {
        "type_name": "Jet",
        "check_attr": "metric_table",
        "skip_fields": "*Idx*G",
    },
    "GenJet": {
        "type_name": "Jet",
        "check_attr": "metric_table",
        "skip_fields": "*Idx*G",
    },
}


@producer(
    uses={
        "Jet.{pt,eta,phi,mass}",
        "GenJet.{pt,eta,phi,mass}",
        attach_coffea_behavior,
    },
    produces={
        "mc_truth_response",
        "mc_truth_response1",
        "mc_truth_responses",
        "MatchedJet.{pt,eta,phi,mass}",
        attach_coffea_behavior,
    },
)
def mc_truth_response(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:

    # attach coffea behaviour
    events = self[attach_coffea_behavior](events, collections=custom_collections, **kwargs)

    # get response for closest Jet to the leading GenJet
    best_match, remaining_jets = delta_r_match(events.GenJet[:, 0], events.Jet, max_dr=0.2, as_index=True)
    response = events.Jet.pt[best_match] / events.GenJet[:, 0].pt
    events = set_ak_column(events, "mc_truth_response", ak.fill_none(ak.nan_to_none(response), EMPTY_FLOAT))

    # get response for the closest Jet to the second leading GenJet
    best_match1, _ = delta_r_match(events.GenJet[:, 1], remaining_jets, max_dr=0.2)
    response1 = ak.singletons(best_match1.pt / events.GenJet[:, 1].pt)
    events = set_ak_column(events, "mc_truth_response1", ak.fill_none(ak.nan_to_none(response1), EMPTY_FLOAT))

    # get responses for the closest Jets to the two leading GenJets
    best_matches, _ = delta_r_match_multiple(events.GenJet[:, :3], events.Jet, max_dr=0.2, as_index=True)
    responses = events.Jet.pt[best_matches] / events.GenJet[:, :3].pt
    events = set_ak_column(events, "mc_truth_responses", ak.fill_none(ak.nan_to_none(responses), EMPTY_FLOAT))

    for field in events.GenJet.fields:
        events = set_ak_column(
            events,
            f"MatchedJet.{field}",
            ak.fill_none(ak.nan_to_none(events.Jet[field][best_matches]), EMPTY_FLOAT),
        )

    return events


@producer(
    uses={
        "Jet.{pt,eta,phi,mass}",
        "GenJet.{pt,eta,phi,mass}",
        attach_coffea_behavior,
    },
    produces={
        "test_var1",
        "test_var2",
    },
)
def test_vars(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections=custom_collections, **kwargs)

    best_matches, _ = delta_r_match_multiple(events.GenJet[:, :2], events.Jet, max_dr=0.2, as_index=True)
    responses = (
        events.Jet.pt[best_matches] * events.GenJet[:, :2].pt
    ) / (events.GenJet[:, :2].pt * events.Jet.pt[best_matches])

    test1 = ak.local_index(responses) * 1.0
    test2 = ak.local_index(responses) * 2.0 + 1
    events = set_ak_column(events, "test_var1", test1)
    events = set_ak_column(events, "test_var2", test2)
    return events
