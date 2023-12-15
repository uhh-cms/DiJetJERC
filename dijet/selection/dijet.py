# coding: utf-8

from typing import Tuple
from columnflow.util import maybe_import
from columnflow.selection import Selector, SelectionResult, selector
from dijet.production.jet_assignment import jet_assignment

ak = maybe_import("awkward")
np = maybe_import("numpy")


@selector(
    uses={
        jet_assignment,
    },
    exposed=True,
)
def dijet_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    # DiJet jet selection
    # - SM or FE
    # - One jet pt>15 GeV
    # - deltaR>2.7

    # dijet

    events = self[jet_assignment](events, **kwargs)

    # two back-to-back leading jets (delta_phi(j1,j2) = min(|phi1 - phi2|, 2PI - |phi2 - phi1|) > 2.7)
    # copied from https://root.cern.ch/doc/master/TVector2_8cxx_source.html#l00103
    # TODO: LorentzVector behavior delta_phi from coffea
    delta_phi = events.probe_jet.phi - events.reference_jet.phi
    delta_phi_pi = ak.where(
        delta_phi >= np.pi,  # condition
        delta_phi - 2 * np.pi,  # if yes
        ak.where(  # else
            delta_phi < ((-1) * np.pi),
            delta_phi + 2 * np.pi,
            delta_phi,
        ),
    )

    dijet_sel = (
        (
            (abs(events.probe_jet.pt) > 15) |
            (abs(events.reference_jet.pt) > 15)
        ) &
        (abs(delta_phi_pi) > 2.7)
    )

    dijet_sel = ak.fill_none(dijet_sel, False)

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={
            "dijet": dijet_sel,
        },
    )
