from typing import Tuple
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.selection import Selector, SelectionResult, selector
from dijet.util import masked_sorted_indices

ak = maybe_import("awkward")


@selector(
    uses={
        "Electron.pt", "Electron.eta",
        "Muon.pt", "Muon.eta",
    },
    produces={"cutflow.n_ele", "cutflow.n_muo"},
    exposed=True,
)
def lepton_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    # lepton selection based on old UHH2 framework
    # https://github.com/UHH2/DiJetJERC/blob/ff98eebbd44931beb016c36327ab174fdf11a83f/src/AnalysisModule_DiJetTrg.cxx#L703
    # IDs in JME Nano https://cms-nanoaod-integration.web.cern.ch/integration/master-106X/mc102X_doc.html
    # mask for muons
    muo_mask = (
        (events.Muon["pt"] > 15) &
        (abs(events.Muon["eta"]) < 2.4) &
        (events.Muon[self.config_inst.x.muon_id])
    )
    # mask for electrons
    ele_mask = (
        (events.Electron["pt"] > 15) &
        (abs(events.Electron["eta"]) < 2.4) &
        (events.Electron[self.config_inst.x.electron_id])
    )

    events = set_ak_column(events, "cutflow.n_ele", ak.sum(ele_mask, axis=1))
    events = set_ak_column(events, "cutflow.n_muo", ak.sum(muo_mask, axis=1))

    # select only events with no leptons
    sel_no_ele = (events.cutflow.n_ele == 0)
    sel_no_muo = (events.cutflow.n_muo == 0)

    ele_indices = masked_sorted_indices(ele_mask, events.Electron["pt"])
    muo_indices = masked_sorted_indices(muo_mask, events.Muon["pt"])

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={
            "no_ele": sel_no_ele,
            "no_muo": sel_no_muo,
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
            "ele_mask": ak.fill_none(ele_mask, False),
            "n_central_electrons": ak.num(ele_indices),
            "muo_mask": ak.fill_none(muo_mask, False),
            "n_central_muons": ak.num(muo_indices),
        },
    )


@lepton_selection.init
def lepton_selection_init(self: Selector) -> None:
    # add lepton ID input columns
    self.uses |= {
        f"Electron.{self.config_inst.x.electron_id}",
        f"Muon.{self.config_inst.x.muon_id}",
    }
