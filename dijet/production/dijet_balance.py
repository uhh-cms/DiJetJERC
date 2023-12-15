# coding: utf-8

"""
Selectors to set ak columns for dijet properties
"""

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production import Producer, producer
from dijet.production.jet_assignment import jet_assignment

np = maybe_import("numpy")
ak = maybe_import("awkward")

"""
Creates column 'Dijet', which includes the most relevant properties of the JetMET dijet analysis.
 - Namely: asymmetry, pt avg of both leading jets and alpha
"""


@producer(
    uses={
        "Jet.pt", "MET.pt", "MET.phi",
        jet_assignment,
    },
    produces={
        "dijets.pt_avg", "dijets.asymmetry", "dijets.mpf", "dijets.mpfx",
    },
)
def dijet_balance(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    # TODO: for now, this only works for reco level
    jets = events.Jet
    jets = ak.pad_none(jets, 3)

    pt_avg = (events.probe_jet.pt + events.reference_jet.pt) / 2
    asym = (events.probe_jet.pt - events.reference_jet.pt) / (2 * pt_avg)
    alpha = jets.pt[:, 2] / pt_avg
    delta_phi = events.probe_jet.phi - events.MET.phi
    mpf = events.MET.pt * np.cos(delta_phi) / (2 * pt_avg)
    mpfx = events.MET.pt * np.sin(delta_phi) / (2 * pt_avg)

    dijets = ak.zip({
        "pt_avg": pt_avg,
        "asymmetry": asym,
        "mpf": mpf,
        "mpfx": mpfx,
    })
    events = set_ak_column(events, "dijets", dijets)

    return events
