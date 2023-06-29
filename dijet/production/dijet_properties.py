# coding: utf-8

"""
Selectors to set ak columns for dijet properties
"""

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column  # , Route, EMPTY_FLOAT
from columnflow.selection import SelectionResult
from columnflow.production import Producer, producer

ak = maybe_import("awkward")

@producer
def dijet_properties(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Creates column 'Dijet', which includes the most relevant properties of the JetMET dijet analysis.
    All sub-fields correspond to single objects with fields pt, eta, phi, mass and pdgId
    or the asymmetry and alpha
    """

    # for quick checks
    def all_or_raise(arr, msg):
        if not ak.all(arr):
            raise Exception(f"{msg} in {100 * ak.mean(~arr):.3f}% of cases")
    
    def debugging():
        from IPython import embed; embed()

    # TODO: for now, this only works for reco level
    jets = events.Jet
    events = set_ak_column(events, "n_jet", ak.num(jets.pt, axis=1), value_type=np.int32)

    # need at least three jets for alpha
    jets = ak.pad_none(jets, 3)
    ghosts = ak.zip({f: EMPTY_FLOAT for f in jets.fields}, with_name="Jet")
    jets = ak.fill_none(jets, ghosts, axis=1)

    pt_avg = jets.pt[0]+jets.pt[1]
    asym = (jets.pt[0]-jets.pt[1])/(2*pt_avg)
    alpha = jets.pt[2]/pt_avg

    events = set_ak_column(events, "pt_avg", pt_avg)
    events = set_ak_column(events, "asymmetry", asym)
    events = set_ak_column(events, "alpha", alpha)

    dijets = {
        "pt_avg": pt_avg,
        "asymmetry": asym,
        "alpha": alpha,
    }
    events = set_ak_column(events, "dijets", dijets)

    return events