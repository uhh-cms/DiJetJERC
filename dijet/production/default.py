# coding: utf-8

"""
Column production methods related to higher-level features.
"""


from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.normalization import normalization_weights
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column

from dijet.production.dijet_balance import dijet_balance
from dijet.production.jet_assignment import jet_assignment
from dijet.production.weights import event_weights


np = maybe_import("numpy")
ak = maybe_import("awkward")

@producer(
    uses={ # features? muon_weights?
        category_ids, normalization_weights,
        dijet_balance, jet_assignment, event_weights,
    },
    produces={ # features?
        category_ids, normalization_weights,
        dijet_balance, jet_assignment, event_weights,
    },
)
def default(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    # category ids
    events = self[category_ids](events, **kwargs)

    # deterministoc seeds
    events = self[category_ids](events, **kwargs)

    # mc-only weights
    if self.dataset_inst.is_mc:
        # normalization weights
        # events = self[normalization_weights](events, **kwargs)
        events = self[event_weights](events, **kwargs)

    # dijet properties: alpha, asymmetry, pt_avg
    # Include MPF production here
    events = self[jet_assignment](events, **kwargs)
    events = self[dijet_balance](events, **kwargs)

    return events
