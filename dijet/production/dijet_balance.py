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
    if self.dataset_inst.is_mc:
        genJets = events.GenJet
    jets = ak.pad_none(jets, 3)

    pt_avg = (events.probe_jet.pt + events.reference_jet.pt) / 2
    if self.dataset_inst.is_mc:

        # check if probe/reference jets have valid gen match

        probe_jet_valid_gen_match = (events.probe_jet.genJetIdx >= 0)
        reference_jet_valid_gen_match = (events.reference_jet.genJetIdx >= 0)
        both_jets_valid_gen_match = (probe_jet_valid_gen_match & reference_jet_valid_gen_match)

        # get gen jet object matched to probe and reference jets
        probe_genJetIdx_mask = ak.singletons(events.probe_jet.genJetIdx)
        reference_genJetIdx_mask = ak.singletons(events.reference_jet.genJetIdx)
        probe_jet_gen_jet = genJets[probe_genJetIdx_mask]
        reference_jet_gen_jet = genJets[reference_genJetIdx_mask]

        response_probe = ak.where(
            probe_jet_valid_gen_match,
            ak.flatten(probe_jet_gen_jet.pt) / events.probe_jet.pt,
            EMPTY_FLOAT,
        )
        response_reference = ak.where(
            reference_jet_valid_gen_match,
            ak.flatten(reference_jet_gen_jet.pt) / events.reference_jet.pt,
            EMPTY_FLOAT,
        )
        # calculate gen-level pT average, MPF, MPFx
        pt_avg_gen = ak.where(
            both_jets_valid_gen_match,
            np.divide(
                ak.flatten(probe_jet_gen_jet.pt) + ak.flatten(reference_jet_gen_jet.pt),
                2,
            ),
            EMPTY_FLOAT,
        )
        mpf_gen = ak.where(
            both_jets_valid_gen_match,
            np.divide(
                events.GenMET.pt * np.cos(ak.flatten(probe_jet_gen_jet.phi) - events.GenMET.phi),
                2 * pt_avg_gen,
            ),
            EMPTY_FLOAT,
        )
        mpfx_gen = ak.where(
            both_jets_valid_gen_match,
            np.divide(
                events.GenMET.pt * np.sin(ak.flatten(probe_jet_gen_jet.phi) - events.GenMET.phi),
                2 * pt_avg_gen,
            ),
            EMPTY_FLOAT,
        )
    asym = (events.probe_jet.pt - events.reference_jet.pt) / (2 * pt_avg)
    alpha = jets.pt[:, 2] / pt_avg
    delta_phi = events.probe_jet.phi - events.MET.phi
    mpf = events.MET.pt * np.cos(delta_phi) / (2 * pt_avg)
    mpfx = events.MET.pt * np.sin(delta_phi) / (2 * pt_avg)
    if self.dataset_inst.is_mc:
        dijets = ak.zip({
            "pt_avg": pt_avg,
            "response_reference": response_reference,
            "response_probe": response_probe,
            "asymmetry": asym,
            "alpha": alpha,
            "mpf": mpf,
            "mpfx": mpfx,
            "mpf_gen": mpf_gen,
            "mpfx_gen": mpfx_gen,
            "pt_avg_gen": pt_avg_gen,
        })
    else:
        dijets = ak.zip({
            "pt_avg": pt_avg,
            "asymmetry": asym,
            "alpha": alpha,
            "mpf": mpf,
            "mpfx": mpfx,
        })

    events = set_ak_column(events, "dijets", dijets)

    return events


@dijet_balance.init
def dijet_balance_init(self: Producer) -> None:
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
            "dijets.mpf_gen",
            "dijets.mpfx_gen",
            "dijets.pt_avg_gen",
        }
