# coding: utf-8

"""
Selection methods defining categories based on selection step results.
"""

from columnflow.util import maybe_import
from columnflow.selection import Selector, selector

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(uses={"event"})
def catid_selection_incl(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    return ak.ones_like(events.event) > 0
