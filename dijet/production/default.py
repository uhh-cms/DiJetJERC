# coding: utf-8

"""
Column production methods related to higher-level features.
"""


from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.normalization import normalization_weights
from columnflow.util import maybe_import

from dijet.production.dijet_balance import dijet_balance
from dijet.production.alpha import alpha
from dijet.production.jet_assignment import jet_assignment
from dijet.production.weights import event_weights

np = maybe_import("numpy")
ak = maybe_import("awkward")


@producer(
    uses={
        category_ids, normalization_weights,
        event_weights,
        dijet_balance, jet_assignment, alpha,
    },
    produces={
        category_ids, normalization_weights,
        event_weights,
        dijet_balance, jet_assignment, alpha,
    },
)
def default(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    # mc-only weights
    if self.dataset_inst.is_mc:
        # normalization weights
        # events = self[normalization_weights](events, **kwargs)
        events = self[event_weights](events, **kwargs)

    # dijet properties: alpha, asymmetry, pt_avg
    # Include MPF production here
    events = self[jet_assignment](events, **kwargs)
    events = self[dijet_balance](events, **kwargs)
    events = self[alpha](events, **kwargs)
    # TODO: Producer for 3rd jet take alpha from dijet_balance

    # category ids
    events = self[category_ids](events, **kwargs)

    # deterministoc seeds
    events = self[category_ids](events, **kwargs)

    return events
