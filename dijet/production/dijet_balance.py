# coding: utf-8

"""
Selectors to set ak columns for dijet properties
"""

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT  # , Route
from columnflow.selection import SelectionResult
from columnflow.production import Producer, producer

np = maybe_import("numpy")
ak = maybe_import("awkward")

@producer(
    uses={"Jet.pt"},
    produces={"dijets.pt_avg","dijets.asymmetry","dijets.alpha"}
)
def dijet_balance(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Creates column 'Dijet', which includes the most relevant properties of the JetMET dijet analysis.
    All sub-fields correspond to single objects with fields pt, eta, phi, mass and pdgId
    or the asymmetry and alpha
    """

    # for quick checks
    def all_or_raise(arr, msg):
        if not ak.all(arr):
            raise Exception(f"{msg} in {100 * ak.mean(~arr):.3f}% of cases")

    # TODO: for now, this only works for reco level
    jets = events.Jet
    events = set_ak_column(events, "n_jet", ak.num(jets.pt, axis=1), value_type=np.int32)
    jets = ak.pad_none(jets, 3)

    pt_avg = (jets.pt[:,0]+jets.pt[:,1])/2
    asym = (jets.pt[:,0]-jets.pt[:,1])/(2*pt_avg)
    alpha = jets.pt[:,2]/pt_avg

    dijets = ak.zip({
        "pt_avg": pt_avg,
        "asymmetry": asym,
        "alpha": alpha,
    })
    events = set_ak_column(events, "dijets", dijets)

    return events