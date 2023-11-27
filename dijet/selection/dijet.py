# coding: utf-8

from typing import Tuple
from columnflow.util import maybe_import
from columnflow.selection import Selector, SelectionResult, selector

ak = maybe_import("awkward")
np = maybe_import("numpy")


@selector(
    uses={
        "use_sm", "use_fe",
        "probe_jet.pt", "probe_jet.eta", "probe_jet.phi", "probe_jet.mass",
        "reference_jet.pt", "reference_jet.eta", "reference_jet.phi", "reference_jet.mass",
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

    # two back-to-back leading jets (delta_phi(j1,j2) = min(|phi1 - phi2|, 2PI - |phi2 - phi1|) > 2.7)
    # copied from https://root.cern.ch/doc/master/TVector2_8cxx_source.html#l00103
    # TODO: LorentzVector behavior delta_phi from coffea
    pi = np.pi
    delta_phi = events.reference_jet.phi - events.probe_jet.phi
    delta_phi_pi = ak.where(
        delta_phi >= pi,  # condition
        delta_phi - 2 * pi,  # if yes
        ak.where(  # else
            delta_phi < ((-1) * pi),
            delta_phi + 2 * pi,
            delta_phi,
        ),
    )

    dijet_sel = (
        (
            (abs(events.probe_jet.pt) > 15) &
            (abs(events.reference_jet.pt) > 15)
        ) &
        (delta_phi_pi > 2.7)
    )

    dijet_sel = ak.fill_none(dijet_sel, False)

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={
            "dijet": dijet_sel,
        },
    )
