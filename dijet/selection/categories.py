# coding: utf-8

"""
Selection methods defining categories based on selection step results.
"""

from columnflow.util import maybe_import
from columnflow.categorization import Categorizer, categorizer

np = maybe_import("numpy")
ak = maybe_import("awkward")


@categorizer(uses={"event"})
def sel_incl(self: Categorizer, events: ak.Array, **kwargs) -> ak.Array:
    return events, ak.ones_like(events.event) > 0
