# coding: utf-8

"""
Column production methods related to higher-level features.
"""


from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.normalization import normalization_weights
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.muon import muon_weights
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column


np = maybe_import("numpy")
ak = maybe_import("awkward")


@producer(
    uses={
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.btagDeepFlavB",
        "dijets",
    },
    produces=set(
        f"cutflow.dijet_{var}"
        for var in ["asymmetry", "alpha", "avg"]
    ),
)
def dijet_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    for var in ["pt", "eta", "phi", "mass"]:
        events = set_ak_column(events, f"cutflow.dijets_{var}", events.dijets[var])

    return events


@producer(
    uses={
        mc_weight, category_ids,
        # nano columns
        "Jet.pt", "Jet.eta", "Jet.phi",
    },
    produces={
        mc_weight, category_ids,
        # new columns
        "cutflow.jet1_pt", "cutflow.n_jets",
    },
)
def cutflow_features(
    self: Producer,
    events: ak.Array,
    object_masks: dict[str, dict[str, ak.Array]],
    **kwargs,
) -> ak.Array:
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    events = self[category_ids](events, **kwargs)

    # apply some per-object selections
    # (here shown for default jets as done in selection.example.jet_selection)
    selected_jet = events.Jet[object_masks["Jet"]["Jet"]]

    events = set_ak_column(events, "cutflow.jet1_pt", Route("pt[:,0]").apply(selected_jet, EMPTY_FLOAT))
    events = set_ak_column(events, "cutflow.n_jets", ak.num(events.Jet.pt, axis=1))

    return events


# @producer(
#     uses={features, category_ids, normalization_weights, muon_weights},
#     produces={features, category_ids, normalization_weights, muon_weights},
# )
# def example(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
#     # features
#     events = self[features](events, **kwargs)

#     # category ids
#     events = self[category_ids](events, **kwargs)

#     # deterministoc seeds
#     events = self[category_ids](events, **kwargs)

#     # mc-only weights
#     if self.dataset_inst.is_mc:
#         # normalization weights
#         events = self[normalization_weights](events, **kwargs)

#         # muon weights
#         events = self[muon_weights](events, **kwargs)

#     return events
