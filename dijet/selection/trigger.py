# coding: utf-8

from typing import Tuple

from columnflow.columnar_util import Route

from columnflow.util import maybe_import
from columnflow.selection import Selector, SelectionResult, selector
from columnflow.production import Producer
from dijet.production.dijet_balance import dijet_balance
from dijet.production.jet_assignment import jet_assignment

from columnflow.selection.empty import empty

ak = maybe_import("awkward")
np = maybe_import("numpy")

@selector(
    uses={
       dijet_balance, jet_assignment,
       "event", "run", "luminosityBlock",
       "HLT.DiPFJetAve*", "HLT.PFJet*",
    },
    exposed=True,
)
def trigger_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:

    # TODO: Work at:
    #       - central vs. forward
    #       - different years
    #       - SingleJet
    #       - Jet collections (AK4 vs. AK8)

    # NOTE: jet_assignment needed when including forward triggers
    events = self[jet_assignment](events, **kwargs)
    events = self[dijet_balance](events, **kwargs)

    # per-event trigger index (based on thresholds)
    if self.dataset_inst.has_tag("missing_dijet_triggers"):
        triggers_central = self.config_inst.x.triggers.singlejet.central
        triggers_forward = self.config_inst.x.triggers.singlejet.central
        thrs_central = self.config_inst.x.trigger_thresholds.singlejet.central
        thrs_forward = self.config_inst.x.trigger_thresholds.singlejet.central
        # NOTE: Keep for later adjustments
        # leading_jet_pt = Route("Jet.pt[:,0]").apply(events, 0.0)
    else:
        triggers_central = self.config_inst.x.triggers.dijet.central
        triggers_forward = self.config_inst.x.triggers.dijet.forward
        thrs_central = self.config_inst.x.trigger_thresholds.dijet.central
        thrs_forward = self.config_inst.x.trigger_thresholds.dijet.forward

    # TODO: In UHH2 singlejet triggers are also checked with pt_avg
    #       Keep for consistency in validation process for now.
    sel_trigger_index_central = np.digitize(ak.to_numpy(events.dijets.pt_avg), thrs_central) - 1  # can be -1
    sel_trigger_index_forward = np.digitize(ak.to_numpy(events.dijets.pt_avg), thrs_forward) - 1  # can be -1

    # mask -1 values to avoid picking wrong trigger. Does not cut events but marks them as unvalid!
    sel_trigger_index_central = ak.mask(sel_trigger_index_central, sel_trigger_index_central < 0, valid_when=False)
    sel_trigger_index_forward = ak.mask(sel_trigger_index_forward, sel_trigger_index_forward < 0, valid_when=False)

    # sanity check: number of thresholds matches the triggers
    assert len(triggers_central) == len(thrs_central)
    assert len(triggers_forward) == len(thrs_forward)

    # put trigger decisions into 2D array
    pass_triggers_central = []
    for trigger_name in triggers_central:
        pass_trigger = getattr(events.HLT, trigger_name)
        pass_triggers_central.append(
            ak.singletons(pass_trigger),
        )
    pass_triggers_central = ak.concatenate(pass_triggers_central, axis=1)

    pass_triggers_forward = []
    for trigger_name in triggers_forward:
        pass_trigger = getattr(events.HLT, trigger_name)
        pass_triggers_forward.append(
            ak.singletons(pass_trigger),
        )
    pass_triggers_forward = ak.concatenate(pass_triggers_forward, axis=1)

    # index; contains none!
    pass_sel_trigger = ak.firsts(
        ak.where(
            abs(events.probe_jet.eta)<2.853,  # TODO: define 2.853 in config
            pass_triggers_central[ak.singletons(sel_trigger_index_central)],
            pass_triggers_forward[ak.singletons(sel_trigger_index_forward)],
        )
    )

    return events, SelectionResult(
        steps={
            "trigger": ak.fill_none(pass_sel_trigger, False),
        },
    )


@trigger_selection.init
def trigger_selection_init(self: Producer) -> None:
    # return immediately if config not yet loaded
    config_inst = getattr(self, "config_inst", None)
    if not config_inst:
        return

    # return immediately if dataset not yet loaded
    dataset_inst = getattr(self, "dataset_inst", None)
    if not dataset_inst:
        return

    # set config dict (use dijet triggers, fall back to single jet if missing)
    if self.dataset_inst.has_tag("missing_dijet_triggers"):
        self.central = self.config_inst.x.triggers.singlejet.central
    else:
        self.central = self.config_inst.x.triggers.dijet.central

    # set input columns
    self.uses |= {
        f"HLT.{var}" for var in self.central
    }
