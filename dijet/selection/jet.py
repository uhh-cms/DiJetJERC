# coding: utf-8

from typing import Tuple
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.selection import Selector, SelectionResult, selector

ak = maybe_import("awkward")


def masked_sorted_indices(mask: ak.Array, sort_var: ak.Array, ascending: bool = False) -> ak.Array:
    """
    Helper function to obtain the correct indices of an object mask
    """
    indices = ak.argsort(sort_var, axis=-1, ascending=ascending)
    return indices[mask[indices]]


@selector(
    uses={"Jet.pt", "Jet.eta", "Jet.phi", "Jet.jetId"},
    produces={"cutflow.n_jet"},
    exposed=True,
)
def jet_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    # DiJet jet selection
    # - require ...

    # assign local index to all Jets - stored after masks for matching
    # TODO: Drop for dijet ?
    events = set_ak_column(events, "Jet.local_index", ak.local_index(events.Jet))

    # jets
    # TODO: Correct jets
    jet_mask = (
        (events.Jet.pt > 25) & (abs(events.Jet.eta) < 5.0) & (events.Jet.jetId == 6)
    )
    events = set_ak_column(events, "cutflow.n_jet", ak.sum(jet_mask, axis=1))
    jet_sel = events.cutflow.n_jet >= 3
    jet_indices = masked_sorted_indices(jet_mask, events.Jet.pt)
    jet_sel = ak.fill_none(jet_sel, False)
    jet_mask = ak.fill_none(jet_mask, False)
    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={
            "Jet": jet_sel,
        },
        objects={
            "Jet": {
                "Jet": jet_indices,
            },
        },
        aux={
            "jet_mask": jet_mask,
            "n_central_jets": ak.num(jet_indices),
        },
    )
