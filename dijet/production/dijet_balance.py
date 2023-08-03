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

@producer(
    uses={
        "Jet.pt", "Jet.eta",
        },
    produces={
        "probe_jet.pt", "probe_jet.eta",
        "reference_jet.pt", "reference_jet.eta",
        "dijets.pt_avg","dijets.asymmetry","dijets.alpha"
        }
)
def dijet_balance(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    # TODO: for now, this only works for reco level
    jets = events.Jet
    events = set_ak_column(events, "n_jet", ak.num(jets.pt, axis=1), value_type=np.int32)
    jets = ak.pad_none(jets, 3)

    """
    Creates column probe and reference jets, used in dijet analysis
        - Probe and referenze jet in same producer as dijet properties
          due to SM and FE method
    """

    # Check for Standard Method
    eta_index_jet1 = np.digitize(ak.to_numpy(jets.eta[:,0]), eta) - 1
    eta_index_jet2 = np.digitize(ak.to_numpy(jets.eta[:,1]), eta) - 1
    isSM = eta_index_jet1 == eta_index_jet2

    # use event numbers in chunk to seed random number generator
    # TODO: use deterministic seeds!
    rand_gen = np.random.Generator(np.random.SFC64(events.event.to_list()))
    rand_ind = ak_random(
        ak.zeros_like(events.event),
        2 * ak.ones_like(events.event),
        rand_func=rand_gen.integers
    )

    # Check for Forward extension
    # We also use FE as a check when both jets are central
    # TODO: not implemented yet!
    leading_is_central = np.abs(jets.eta[:,0]) < 1.305
    subleading_is_central = np.abs(jets.eta[:,1]) < 1.305
    isFE = leading_is_central ^ subleading_is_central # exclusive or

    # Avoid ak.where: if leading is central jet the prob jet is the subleading one
    pro_index = isFE * leading_is_central + (1 - isFE) * (rand_ind != 0)
    ref_index = isFE * (1-leading_is_central) + (1 - isFE) * (rand_ind == 0)

    # Assign jets to probe and reference
    probe_jet = ak.where(pro_index == 0, jets[:, 0], jets[:, 1])
    reference_jet = ak.where(pro_index == 1, jets[:, 1], jets[:, 0])

    # Only produce pt and eta fields
    probe_jet = ak.zip({ "pt": probe_jet.pt,  "eta": probe_jet.eta })
    events = set_ak_column(events, "probe_jet", reference_jet)

    reference_jet = ak.zip({ "pt": reference_jet.pt,  "eta": reference_jet.eta })
    events = set_ak_column(events, "reference_jet", reference_jet)

    """
    Creates column 'Dijet', which includes the most relevant properties of the JetMET dijet analysis.
    All sub-fields correspond to single objects with fields pt, eta, phi, mass and pdgId
    or the asymmetry and alpha
    """

    # TODO: Asymmetry of probe_jet in FE method dependent of SM method
    #       Not yet implemented!
    pt_avg = (probe_jet.pt+reference_jet.pt)/2
    asym = (probe_jet.pt-reference_jet.pt)/(2*pt_avg)
    alpha = jets.pt[:,2]/pt_avg

    dijets = ak.zip({
        "pt_avg": pt_avg,
        "asymmetry": asym,
        "alpha": alpha,
        "SM": isSM,
        "FE": isFE,
    })
    events = set_ak_column(events, "dijets", dijets)

    return events