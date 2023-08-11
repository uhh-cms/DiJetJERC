from typing import Tuple
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.selection import Selector, SelectionResult, selector

ak = maybe_import("awkward")


def masked_sorted_indices(mask: ak.Array, sort_var: ak.Array, ascending: bool = False) -> ak.Array:
    """
    Helper function to obtain the correct indices of an object mask
    """
    indices = ak.argsort(sort_var, axis=-1, ascending=ascending)
    return indices[mask[indices]]


@selector(

    uses={"Electron.pt", "Electron.eta", "Muon.pt", "Muon.eta"},
    produces={"cutflow.n_ele", "cutflow.n_muo"},
    exposed=True,
)
def lepton_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    events = set_ak_column(events, "local_index", ak.local_index(events.Jet))

    # mask for muons
    muo_mask = (
        (events.Muon.pt > 20) & (abs(events.Muon.eta) < 2.4)
    )
    # mask for electrons
    ele_mask = (
        (events.Electron.pt > 20) & (abs(events.Electron.eta) < 2.4)
    )

    events = set_ak_column(events, "cutflow.n_ele", ak.sum(ele_mask, axis=1))
    events = set_ak_column(events, "cutflow.n_muo", ak.sum(muo_mask, axis=1))
    ele_sel = events.cutflow.n_ele == 0
    muo_sel = events.cutflow.n_muo == 0

    # select only events with no leptons

    lep_sel = (events.cutflow.n_ele == 0) & (events.cutflow.n_muo == 0)

    ele_indices = masked_sorted_indices(ele_mask, events.Electron.pt)
    muo_indices = masked_sorted_indices(muo_mask, events.Muon.pt)

    ele_sel = ak.fill_none(ele_sel, False)
    ele_mask = ak.fill_none(ele_mask, False)

    muo_sel = ak.fill_none(muo_sel, False)
    muo_mask = ak.fill_none(muo_mask, False)

    lep_sel = ak.fill_none(lep_sel, False)

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={
            "Lepton": lep_sel,
        },
        objects={
            "Electron": {
                "Electron": ele_indices,
            },
            "Muon": {
                "Muon": muo_indices,
            },
        },
        aux={
            "ele_mask": ele_mask,
            "n_central_eletons": ak.num(ele_indices),
            "muo_mask": muo_mask,
            "n_central_muons": ak.num(muo_indices),
        },
    )
