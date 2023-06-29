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
    uses={"Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.btagDeepFlavB", "Jet.jetId"},
    produces={"cutflow.n_jet", "cutflow.n_deepjet_med"},
    exposed=True,
)
def jet_selection(
    self: Selector,
    events: ak.Array,
    lepton_results: SelectionResult,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    # dijet jet selection
    # - Write down  requirment

    # assign local index to all Jets
    dummy = 0


@selector(
    uses={
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass",
        "Electron.cutBased", "Electron.mvaFall17V2Iso_WP80",
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass",
        "Muon.tightId", "Muon.looseId", "Muon.pfRelIso04_all",
        "Tau.pt", "Tau.eta", "Tau.idDeepTau2017v2p1VSe",
        "Tau.idDeepTau2017v2p1VSmu", "Tau.idDeepTau2017v2p1VSjet",
    },
    e_pt=None, mu_pt=None, e_trigger=None, mu_trigger=None,
)
def lepton_selection(
        self: Selector,
        events: ak.Array,
        stats: defaultdict,
        **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    # dijet lepton selection
    # - require exactly 0 leptons
    dummy = 0



@selector(
    uses={
        category_ids, process_ids, attach_coffea_behavior,
        "mc_weight",  # not opened per default but always required in Cutflow tasks
    },
    produces={
        category_ids, process_ids, attach_coffea_behavior,
        "mc_weight",  # not opened per default but always required in Cutflow tasks
    },
    exposed=True,
)
def default(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
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