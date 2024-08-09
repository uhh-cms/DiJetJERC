# coding: utf-8

"""
Selectors to set ak columns for dijet properties
"""

from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT
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

    # ensure at least three jets
    jets = ak.pad_none(jets, 3)

    # compute derived quantities
    pt_avg = (events.probe_jet.pt + events.reference_jet.pt) / 2
    asym = (events.probe_jet.pt - events.reference_jet.pt) / (2 * pt_avg)
    alpha = jets.pt[:, 2] / pt_avg
    delta_phi = events.probe_jet.phi - events.MET.phi
    mpf = events.MET.pt * np.cos(delta_phi) / (2 * pt_avg)
    mpfx = events.MET.pt * np.sin(delta_phi) / (2 * pt_avg)

    # array to return (filling in missing values)
    dijets = {
        "pt_avg": ak.fill_none(pt_avg, EMPTY_FLOAT),
        "asymmetry": ak.fill_none(asym, EMPTY_FLOAT),
        "alpha": ak.fill_none(alpha, EMPTY_FLOAT),
        "mpf": ak.fill_none(mpf, EMPTY_FLOAT),
        "mpfx": ak.fill_none(mpfx, EMPTY_FLOAT),
    }

    # add MC-specific things
    if self.dataset_inst.is_mc:
        # gen level jets
        gen_jets = events.GenJet

        # check if probe/reference jets have valid gen match
        probe_jet_valid_gen_match = (events.probe_jet.genJetIdx >= 0)
        reference_jet_valid_gen_match = (events.reference_jet.genJetIdx >= 0)
        both_jets_valid_gen_match = (probe_jet_valid_gen_match & reference_jet_valid_gen_match)

        # get gen jet object matched to probe and reference jets
        probe_genJetIdx_mask = ak.singletons(events.probe_jet.genJetIdx)
        reference_genJetIdx_mask = ak.singletons(events.reference_jet.genJetIdx)
        probe_jet_gen_jet = ak.firsts(gen_jets[probe_genJetIdx_mask])
        reference_jet_gen_jet = ak.firsts(gen_jets[reference_genJetIdx_mask])

        # gen-level probe jet response
        response_probe = ak.mask(
            probe_jet_gen_jet.pt / events.probe_jet.pt,
            mask=probe_jet_valid_gen_match,
            valid_when=True,
        )

        # gen-level reference jet response
        response_reference = ak.mask(
            reference_jet_gen_jet.pt / events.reference_jet.pt,
            mask=reference_jet_valid_gen_match,
            valid_when=True,
        )

        # gen-level average pT
        pt_avg_gen = ak.mask(
            np.divide(
                probe_jet_gen_jet.pt + reference_jet_gen_jet.pt,
                2,
            ),
            mask=both_jets_valid_gen_match,
            valid_when=True,
        )

        # gen-level asymmetry
        asymmetry_gen = ak.mask(
            np.divide(
                (probe_jet_gen_jet.pt - reference_jet_gen_jet.pt),
                2 * pt_avg_gen,
            ),
            mask=both_jets_valid_gen_match,
            valid_when=True,
        )

        # gen-level MPF
        mpf_gen = ak.mask(
            np.divide(
                events.GenMET.pt * np.cos(probe_jet_gen_jet.phi - events.GenMET.phi),
                2 * pt_avg_gen,
            ),
            mask=both_jets_valid_gen_match,
            valid_when=True,
        )

        # gen-level MPFx
        mpfx_gen = ak.mask(
            np.divide(
                events.GenMET.pt * np.sin(probe_jet_gen_jet.phi - events.GenMET.phi),
                2 * pt_avg_gen,
            ),
            mask=both_jets_valid_gen_match,
            valid_when=True,
        )

        # update return array
        dijets.update({
            "response_reference": ak.fill_none(response_reference, EMPTY_FLOAT),
            "response_probe": ak.fill_none(response_probe, EMPTY_FLOAT),
            "asymmetry_gen": ak.fill_none(asymmetry_gen, EMPTY_FLOAT),
            "mpf_gen": ak.fill_none(mpf_gen, EMPTY_FLOAT),
            "mpfx_gen": ak.fill_none(mpfx_gen, EMPTY_FLOAT),
            "pt_avg_gen": ak.fill_none(pt_avg_gen, EMPTY_FLOAT),
        })

    # write out return array
    events = set_ak_column(events, "dijets", ak.zip(dijets))

    return events


@dijet_balance.init
def dijet_balance_init(self: Producer) -> None:
    if not getattr(self, "dataset_inst", None):
        return
    if self.dataset_inst.is_mc:
        self.uses |= {
            "GenJet.pt",
            "GenJet.phi",
            "probe_jet.genJetIdx",
            "reference_jet.genJetIdx",
            "GenMET.pt",
            "GenMET.phi",
        }

        self.produces |= {
            "dijets.response_probe",
            "dijets.response_reference",
            "dijets.asymmetry_gen",
            "dijets.mpf_gen",
            "dijets.mpfx_gen",
            "dijets.pt_avg_gen",
        }
