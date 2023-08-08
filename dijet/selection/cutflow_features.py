# coding: utf-8

"""
Column production methods related to higher-level features.
"""


from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column

np = maybe_import("numpy")
ak = maybe_import("awkward")


@producer(
    uses={
        mc_weight, category_ids,
        # nano columns
        "Jet.pt", "Jet.eta", "Jet.phi",
        "dijets.asymmetry", "dijets.alpha", "dijets.pt_avg",
    },
    produces={
        mc_weight, category_ids,
        # new columns
        "cutflow.jet1_pt", "cutflow.n_jets",
    } | {
        f"cutflow.dijets_{var}"
        for var in ["asymmetry", "alpha", "pt_avg"]
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

    for var in ["pt_avg", "asymmetry", "alpha"]:
        events = set_ak_column(events, f"cutflow.dijets_{var}", events.dijets[var])

    return events
