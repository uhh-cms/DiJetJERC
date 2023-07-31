# coding: utf-8

"""
Selection methods for HHtobbWW.
"""

from collections import defaultdict
from typing import Tuple

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.util import attach_coffea_behavior

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.production.categories import category_ids
from columnflow.production.processes import process_ids

# from dijet.production.weights import event_weights_to_normalize
from dijet.production.dijet_properties import dijet_properties
from dijet.selection.dijet import dijet_features

np = maybe_import("numpy")
ak = maybe_import("awkward")

def masked_sorted_indices(mask: ak.Array, sort_var: ak.Array, ascending: bool = False) -> ak.Array:
    """
    Helper function to obtain the correct indices of an object mask
    """
    indices = ak.argsort(sort_var, axis=-1, ascending=ascending)
    return indices[mask[indices]]

@selector(
    uses={
        category_ids, process_ids, attach_coffea_behavior,
        mc_weight, large_weights_killer,  # not opened per default but always required in Cutflow tasks
        jet_selection, dijet_balance, cutflow_features, dijet_increment_stats,
    },
    produces={
        category_ids, process_ids, attach_coffea_behavior,
        mc_weight, large_weights_killer,
        jet_selection, dijet_balance, cutflow_features, dijet_increment_stats,
    },
    exposed=True,
    check_used_columns=False,
    check_produced_columns=False,
)
def default(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:

    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)
        events = self[large_weights_killer](events, **kwargs)

    # ensure coffea behavior
    events = self[attach_coffea_behavior](events, **kwargs)

    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # # TODO Implement selection
    # # lepton selection
    # events, lepton_results = self[lepton_selection](events, stats, **kwargs)
    # results += lepton_results

    # # jet selection
    # events, jet_results = self[jet_selection](events, lepton_results, stats, **kwargs)
    # results += jet_results

    # combined event selection after all steps
    # NOTE: we only apply the b-tagging step when no AK8 Jet is present; if some event with AK8 jet
    #       gets categorized into the resolved category, we might need to cut again on the number of b-jets
    # results.main["event"] = (
    #     results.steps.all_but_bjet &
    #     ((results.steps.Jet & results.steps.Bjet) | results.steps.Boosted)
    # )

    # build categories
    events = self[category_ids](events, results=results, **kwargs)

    # create process ids
    events = self[process_ids](events, **kwargs)

    return events, results


# @default.init
# def default_init(self: Selector) -> None:
#     if self.config_inst.x("do_cutflow_features", False):
#         self.uses.add(cutflow_features)
#         self.produces.add(cutflow_features)

#     if not getattr(self, "dataset_inst", None) or self.dataset_inst.is_data:
#         return

#     self.uses.add(event_weights_to_normalize)
#     self.produces.add(event_weights_to_normalize)


@selector(
    uses={
        default, "mc_weight",  # mc_weight should be included from default
        dijet_properties, dijet_features,
    },
    produces={
        category_ids, process_ids, "mc_weight",
        dijet_properties, dijet_features,
    },
    exposed=True,
)
def dijet_selection(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    """
    Selector that is used to perform dijet studies
    """
    # from IPython import embed; embed();
    # run the default Selector
    events, results = self[default](events, stats, **kwargs)

    # extract relevant dijet variables
    events = self[dijet_properties](events, **kwargs)

    # produce relevant columns
    events = self[dijet_features](events, **kwargs)

    return events, results