# coding: utf-8

"""
Selectors to set ak columns for dijet properties
"""

# TODO: Not all columns needed are present yet for pt balance method
#       - Include probe and reference jet from jet assignment
#       - is FE or SM Method

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production import Producer, producer

np = maybe_import("numpy")
ak = maybe_import("awkward")

"""
Creates column 'Dijet', which includes the most relevant properties of the JetMET dijet analysis.
 - Namely: asymmetry, pt avg of both leading jets and alpha
"""


@producer(
    uses={
        "Jet.pt",
        "probe_jet.pt", "reference_jet.pt", "probe_jet.phi",
        "MET.pt", "MET.phi",
    },
    produces={
        "dijets.pt_avg", "dijets.asymmetry", "dijets.alpha", "dijets.mpf", "dijets.mpfx",
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
        probe_genJet_pt = np.array(
            [
                all_genJets[index] for all_genJets, index in zip(genJets.pt, events.probe_jet.genJetIdx)
            ],
        )
        reference_genJet_pt = np.array(
            [
                all_genJets[index] for all_genJets, index in zip(genJets.pt, events.reference_jet.genJetIdx)
            ],
        )
        probe_genJet_phi = np.array(
            [
                all_genJets[index] for all_genJets, index in zip(genJets.phi, events.probe_jet.genJetIdx)
            ],
        )
        response_probe = ak.where(
            events.probe_jet.genJetIdx != -1, probe_genJet_pt / events.probe_jet.pt, -100,
        )
        response_reference = ak.where(
            events.reference_jet.genJetIdx != -1, reference_genJet_pt / events.reference_jet.pt, -100,
        )
        refAndProbe = (events.reference_jet.genJetIdx != -1) & (events.probe_jet.genJetIdx != -1)
        pt_avg_gen = ak.where(
            refAndProbe, np.divide((probe_genJet_pt + reference_genJet_pt), 2), -100,
        )
        mpf_gen_val = np.divide(events.GenMET.pt * np.cos(probe_genJet_phi - events.GenMET.phi), (2 * pt_avg_gen))
        mpf_gen = ak.where(
            refAndProbe, mpf_gen_val, -100,
        )
        mpfx_gen_val = np.divide(events.GenMET.pt * np.sin(probe_genJet_phi - events.GenMET.phi), (2 * pt_avg_gen))
        mpfx_gen = ak.where(
            refAndProbe, mpfx_gen_val, -100,
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
        self.uses |= {"GenJet.pt", "GenJet.phi", "probe_jet.genJetIdx",
        "reference_jet.genJetIdx", "GenMET.pt", "GenMET.phi"}
        self.produces |= {"dijets.response_probe", "dijets.response_reference",
        "dijets.mpf_gen", "dijets.mpfx_gen", "dijets.pt_avg_gen"}
