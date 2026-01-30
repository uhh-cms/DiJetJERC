# coding: utf-8

"""
Column production methods related to generic event weights.
"""

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.selection import SelectionResult
from columnflow.production import Producer, producer
from columnflow.production.cms.pileup import pu_weight
from columnflow.production.normalization import normalization_weights

np = maybe_import("numpy")
ak = maybe_import("awkward")



@producer(
    uses={
        pu_weight
    },
    produces={
        pu_weight
    },
    mc_only=True,
)
def event_weights_to_normalize(self: Producer, events: ak.Array, results: SelectionResult, **kwargs) -> ak.Array:
    """
    Wrapper of several event weight producers that are typically called as part of SelectEvents
    since it is required to normalize them before applying certain event selections.
    """

    # compute pu weights
    events = self[pu_weight](events, **kwargs)

    return events


@event_weights_to_normalize.init
def event_weights_to_normalize_init(self) -> None:
    pass


@producer(
    uses={
        normalization_weights,
        pu_weight,
    },
    produces={
        normalization_weights,
        pu_weight
    },
    mc_only=True,
)
def event_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Wrapper of several event weight producers that are typically called in ProduceColumns.
    """

    # compute normalization weights
    events = self[normalization_weights](events, **kwargs)
    # compute pileup weights
    events = self[pu_weight](events, **kwargs)

    return events


@event_weights.init
def event_weights_init(self: Producer) -> None:
    pass


@producer(
    uses={"mc_weight"},
    produces={"mc_weight"},
    mc_only=True,
)
def large_weights_killer(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Simple producer that sets eventweights to 0 when too large.
    """
    if self.dataset_inst.is_data:
        raise Exception("large_weights_killer is only callable for MC")

    # TODO: figure out a good threshold when events are considered unphysical
    median_weight = ak.sort(abs(events.mc_weight))[int(len(events) / 2)]
    weight_too_large = abs(events.mc_weight) > 1000 * median_weight
    events = set_ak_column(events, "mc_weight", ak.where(weight_too_large, 0, events.mc_weight))

    return events
