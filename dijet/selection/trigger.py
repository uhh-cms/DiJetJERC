# coding: utf-8

from typing import Tuple
from columnflow.util import maybe_import
from columnflow.selection import Selector, SelectionResult, selector
from columnflow.production import Producer
from collections import defaultdict

ak = maybe_import("awkward")
np = maybe_import("numpy")


def masked_sorted_indices(mask: ak.Array, sort_var: ak.Array, ascending: bool = False) -> ak.Array:
    """
    Helper function to obtain the correct indices of an object mask
    """
    indices = ak.argsort(sort_var, axis=-1, ascending=ascending)
    return indices[mask[indices]]


@selector(
    uses={
        "dijets.pt_avg", "probe_jet.eta",
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


    # per-event trigger index (based on thresholds)
    thrs = self.config_inst.x.trigger_thresholds.dijet.central
    sel_trigger_index = np.digitize(ak.to_numpy(events.dijets.pt_avg), thrs) - 1  # can be -1

    # mask -1 values to avoid picking wrong trigger. Does not cut events but marks them as unvalid!
    sel_trigger_index = ak.mask(sel_trigger_index, sel_trigger_index < 0, valid_when=False)

    # sanity check: number of thresholds matches the triggers
    trigger_names = self.config_inst.x.triggers.dijet.central
    assert len(trigger_names) == len(thrs)

    # put trigger decisions into 2D array
    pass_triggers = []
    for trigger_name in self.config_inst.x.triggers.dijet.central:
        pass_trigger = getattr(events.HLT, trigger_name)
        pass_triggers.append(
            ak.singletons(pass_trigger),
        )
    pass_triggers = ak.concatenate(pass_triggers, axis=1)

    # index; contains none!
    pass_sel_trigger = ak.firsts(pass_triggers[ak.singletons(sel_trigger_index)])

    return events, SelectionResult(
        steps={
            "trigger": ak.fill_none(pass_sel_trigger, False),
        },
    )


@trigger_selection.init
def trigger_selection_init(self: Selector) -> None:
    # return immediately if config not yet loaded
    config_inst = getattr(self, "config_inst", None)
    if not config_inst:
        return

    dataset_inst = getattr(self, "dataset_inst", None)
    if not dataset_inst:
        return

    switch_trigger_UL17 = self.config_inst.campaign.x.year == 2017 and self.dataset_inst.x.era in ["B", "C"]

    # set config dict
    self.central = (
        self.config_inst.x.triggers.singlejet.central
        if switch_trigger_UL17
        else
        self.config_inst.x.triggers.dijet.central
    )
    self.forward = (
        self.config_inst.x.triggers.singlejet.central
        if switch_trigger_UL17
        else
        self.config_inst.x.triggers.dijet.forward
    )

    # set input columns
    self.uses |= {
        f"HLT.{var}" for var in self.central
    } | {
        f"HLT.{var}" for var in self.forward
    }
