# coding: utf-8

from typing import Tuple

from columnflow.columnar_util import Route

from columnflow.util import maybe_import
from columnflow.selection import Selector, SelectionResult, selector
from columnflow.production import Producer

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
    if self.dataset_inst.has_tag("missing_dijet_triggers"):
        trigger_names = self.config_inst.x.triggers.singlejet.central
        thrs = self.config_inst.x.trigger_thresholds.singlejet.central
        leading_jet_pt = Route("Jet.pt[:,0]").apply(events, 0.0)
        sel_trigger_index = np.digitize(ak.to_numpy(leading_jet_pt), thrs) - 1  # can be -1
    else:
        trigger_names = self.config_inst.x.triggers.dijet.central
        thrs = self.config_inst.x.trigger_thresholds.dijet.central
        sel_trigger_index = np.digitize(ak.to_numpy(events.dijets.pt_avg), thrs) - 1  # can be -1

    # mask -1 values to avoid picking wrong trigger. Does not cut events but marks them as unvalid!
    sel_trigger_index = ak.mask(sel_trigger_index, sel_trigger_index < 0, valid_when=False)

    # sanity check: number of thresholds matches the triggers
    assert len(trigger_names) == len(thrs)

    # put trigger decisions into 2D array
    pass_triggers = []
    for trigger_name in trigger_names:
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
