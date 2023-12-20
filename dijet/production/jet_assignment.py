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
from columnflow.calibration.util import ak_random

from dijet.constants import eta

np = maybe_import("numpy")
ak = maybe_import("awkward")

"""
Creates column probe and reference jets, used in dijet analysis.
Only store pt, eta and phi field for now and remove mass and pdgId
"""


@producer(
    uses={
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass",
    },
    produces={
        "use_sm", "use_fe", "n_jet",
        "probe_jet.pt", "probe_jet.eta", "probe_jet.phi", "probe_jet.mass",
        "reference_jet.pt", "reference_jet.eta", "reference_jet.phi", "reference_jet.mass",
    },
)
def jet_assignment(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    # TODO: for now, this only works for reco level
    jets = events.Jet
    events = set_ak_column(events, "n_jet", ak.num(jets.pt, axis=1), value_type=np.int32)
    jets = ak.pad_none(jets, 2)

    # Check for Standard Method
    eta_index_jet1 = np.digitize(ak.to_numpy(jets.eta[:, 0]), eta) - 1
    eta_index_jet2 = np.digitize(ak.to_numpy(jets.eta[:, 1]), eta) - 1
    use_sm = eta_index_jet1 == eta_index_jet2
    events = set_ak_column(events, "use_sm", use_sm)

    # use event numbers in chunk to seed random number generator
    # TODO: use deterministic seeds!
    rand_gen = np.random.Generator(np.random.SFC64(events.event.to_list()))
    rand_ind = ak_random(
        ak.zeros_like(events.event),
        2 * ak.ones_like(events.event),
        rand_func=rand_gen.integers,
    )

    # Check for Forward extension
    # We also use FE as a check when both jets are central
    # TODO: not implemented yet!
    leading_is_central = np.abs(jets.eta[:, 0]) < 1.305
    subleading_is_central = np.abs(jets.eta[:, 1]) < 1.305
    use_fe = leading_is_central ^ subleading_is_central  # exclusive or
    events = set_ak_column(events, "use_fe", use_fe)

    # index of the central jet
    central_ind = ak.values_astype(leading_is_central, np.uint8)

    # if FE, choose the central jet as the probe, otherwise random
    pro_index = ak.where(use_fe, central_ind, rand_ind)
    ref_index = ak.where(use_fe, 1 - central_ind, 1 - rand_ind)

    # Assign jets to probe and reference
    probe_jet = ak.firsts(jets[ak.singletons(pro_index)])
    reference_jet = ak.firsts(jets[ak.singletons(ref_index)])

    # write out probe and reference jet fields
    fields = ("pt", "eta", "phi", "mass", "genJetIdx") if self.dataset_inst.is_mc else ("pt", "eta", "phi", "mass")
    probe_jet = ak.zip({f: getattr(probe_jet, f) for f in fields})
    events = set_ak_column(events, "probe_jet", probe_jet)

    reference_jet = ak.zip({f: getattr(reference_jet, f) for f in fields})
    events = set_ak_column(events, "reference_jet", reference_jet)

    return events


@jet_assignment.init
def jet_assignment_init(self: Producer) -> None:
    if self.dataset_inst.is_mc:
        self.uses |= {"Jet.genJetIdx"}
        self.produces |= {"reference_jet.genJetIdx", "probe_jet.genJetIdx"}
