# coding: utf-8
from __future__ import annotations

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


@producer(
    uses={
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass",
    },
    produces={
        "use_sm", "use_fe",
        "probe_jet.pt", "probe_jet.eta", "probe_jet.phi", "probe_jet.mass",
        "reference_jet.pt", "reference_jet.eta", "reference_jet.phi", "reference_jet.mass",
    },
)
def jet_assignment(
    self: Producer,
    events: ak.Array,
    # mask to apply to `event.Jet` before running assignment
    jet_mask: ak.Array | None = None,
    **kwargs
) -> ak.Array:
    """
    Producer to assign the probe and reference jets. Creates new four-vector
    columns `probe_jet` and `reference_jet` as main result. Also produces
    flags `use_sm` and `use_fe`, indicating whether the Standard Method (SM)
    or Forward Extension (FE) method should be used for JER derivation later on.

    The probe and reference jet assignment proceeds as follows:

    If Jet[0] and Jet[1] are in the same |eta| bin, the SM is used and the jets
    are assigned randomly as the probe and reference jets.

    Otherwise, if either one of Jet[0] or Jet[1] lies in the central region
    (|eta| < 1.131), it is selected as the reference jet, while the other one
    is considered the probe jet. The FE method is used in this case.
    """

    # retrieve jet collection
    jets = events.Jet

    # apply jet mask, if provided
    if jet_mask is not None:
        jets = jets[jet_mask]

    # pad jet collection to at least length two
    jets = ak.pad_none(jets, 2)

    # retrieve absolute eta of two leading jets
    abseta_jet1 = np.abs(jets.eta[:, 0])
    abseta_jet2 = np.abs(jets.eta[:, 1])

    # check if two leading jets are in the same |eta| bin
    abseta_index_jet1 = np.digitize(ak.to_numpy(abseta_jet1), eta) - 1
    abseta_index_jet2 = np.digitize(ak.to_numpy(abseta_jet2), eta) - 1
    jets_same_abseta_bin = (abseta_index_jet1 == abseta_index_jet2)

    # set a flag for when the Standard Method should be used
    events = set_ak_column(events, "use_sm", jets_same_abseta_bin)

    # use event numbers in chunk to seed random number generator
    # TODO: use deterministic seeds
    rand_gen = np.random.Generator(np.random.SFC64(events.event.to_list()))
    rand_ind = ak_random(
        ak.zeros_like(events.event),
        2 * ak.ones_like(events.event),
        rand_func=rand_gen.integers,
    )

    # check whether leading jets are in the central detector region
    jet1_is_central = (abseta_jet1 < 1.131)
    jet2_is_central = (abseta_jet2 < 1.131)
    both_jets_central = (jet1_is_central & jet2_is_central)
    atleast_one_jet_central = (jet1_is_central | jet2_is_central)

    # set a flag for when the Forward Extension should be used
    # (at least one jet in the central region, except if if is in
    # the same |eta| bin as the other jet -> SM)
    use_fe = (
        atleast_one_jet_central &
        ~jets_same_abseta_bin
    )
    events = set_ak_column(events, "use_fe", use_fe)

    # determine index of central jet
    idx_central_jet = ak.values_astype(
        ak.where(
            jet1_is_central,
            0,  # leading jet is central jet -> index 0
            1,  # subleading jet is central jet -> index 1
        ),
        np.uint8,
    )

    # compute index of reference and probe jets:
    # - if one jet central and the other noncentral -> pick central jet as reference
    # - if both or neither jet central -> pick randomly
    ref_index = ak.where(
        atleast_one_jet_central & ~both_jets_central,
        idx_central_jet,
        rand_ind,
    )
    pro_index = 1 - ref_index  # the opposite

    # Assign jets to probe and reference
    probe_jet = ak.firsts(jets[ak.singletons(pro_index)])
    reference_jet = ak.firsts(jets[ak.singletons(ref_index)])

    # write out probe and reference jet fields
    fields = ("pt", "eta", "phi", "mass")
    if self.dataset_inst.is_mc:
        fields += ("genJetIdx",)
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
