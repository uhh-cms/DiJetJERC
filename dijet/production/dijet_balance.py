# coding: utf-8

"""
Selectors to set ak columns for dijet properties
"""

# TODO: Not all columns needed are present yet for pt balance method
#       - Include probe and reference jet from jet assignment
#       - is FE or SM Method

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT  # , Route
from columnflow.selection import SelectionResult
from columnflow.production import Producer, producer
from columnflow.calibration.util import ak_random

from dijet.constants import eta

np = maybe_import("numpy")
ak = maybe_import("awkward")

"""
Creates column 'Dijet', which includes the most relevant properties of the JetMET dijet analysis.
 - Namely: asymmetry, pt avg of both leading jets and alpha
"""

@producer(
    uses={
        "Jet.pt",
        "probe_jet.pt", "reference_jet.pt"
        },
    produces={
        "dijets.pt_avg","dijets.asymmetry","dijets.alpha"
        }
)
def dijet_balance(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    # TODO: for now, this only works for reco level
    jets = events.Jet
    jets = ak.pad_none(jets, 3)

    pt_avg = (events.probe_jet.pt+events.reference_jet.pt)/2
    asym = (events.probe_jet.pt-events.reference_jet.pt)/(2*pt_avg)
    alpha = jets.pt[:,2]/pt_avg

    dijets = ak.zip({
        "pt_avg": pt_avg,
        "asymmetry": asym,
        "alpha": alpha,
    })
    events = set_ak_column(events, "dijets", dijets)

    return events