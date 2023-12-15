# coding: utf-8

"""
Producer to set ak columns for alpha
"""

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production import Producer, producer
from dijet.production.dijet_balance import dijet_balance

np = maybe_import("numpy")
ak = maybe_import("awkward")

"""
Creates column 'alpha'.
"""


@producer(
    uses={
        "Jet.pt",
        dijet_balance,
    },
    produces={
        "alpha",
    },
)
def alpha(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    events = self[dijet_balance](events, **kwargs)

    # TODO: for now, this only works for reco level
    jets = events.Jet
    jets = ak.pad_none(jets, 3)

    alpha = ak.where(
        jets.pt[:, 2]>15,
        jets.pt[:, 2] / events.dijets.pt_avg,
        ak.where(
            ak.num(events.Jet)<3,
            0,
            1,
        ),
    )
    events = set_ak_column(events, "alpha", alpha)

    return events
