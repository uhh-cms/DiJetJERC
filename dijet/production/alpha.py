# coding: utf-8

"""
Producer to set ak columns for alpha
"""
from __future__ import annotations

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production import Producer, producer
from dijet.production.dijet_balance import dijet_balance

np = maybe_import("numpy")
ak = maybe_import("awkward")


@producer(
    uses={dijet_balance},
    # name of the jet collection to run on
    jet_name="Jet",
    # name of the output column
    alpha_name="alpha",
)
def alpha(
    self: Producer,
    events: ak.Array,
    # mask to apply to `event.Jet` before computing alpha
    jet_mask: ak.Array | None = None,
    #  pT for a jet to be considered when computing alpha
    jet3_min_pt: float = 15.,
    **kwargs
) -> ak.Array:
    """
    Computes the quantity 'alpha', which measures the additional jet activity in the event,
    and is, in general, defined as the ratio of the third-leading jet pT to the average pT of
    the two leading jets.

    This definition is altered when the third-leading jet
    The 'alpha' value computed by this producer follows the definition in UHH2,
    """

    # retrieve jet collection
    jets = events[self.jet_name]

    # apply jet mask, if provided
    if jet_mask is not None:
        jets = jets[jet_mask]

    # compute number of jets
    n_jet = ak.num(jets, axis=1)

    # pad jets to ensure there are at least three,
    # inserting 'None' as needed
    jets_padded = ak.pad_none(jets, 3)

    # compute raw alpha
    alpha_raw = jets_padded["pt"][:, 2] / events.dijets.pt_avg

    # refine alpha computation given pT threshold
    # - if third jet is above threshold -> use alpha_raw unchanged
    # - else, check number of jets passing selection mask (n_jet)
    #   - if n_jet <= 2: set alpha to 0
    #   - else: set alpha to 1

    # check if third jet is above threshold A
    jet3_is_above_thr = ak.fill_none(
        jets_padded["pt"][:, 2] > jet3_min_pt,
        False,
    )

    # compute alpha
    alpha = ak.where(
        jet3_is_above_thr,
        alpha_raw,
        ak.where(
            n_jet <= 2,
            0,
            # note: better agreement with UHH2 if using `alpha_raw`, but seems to contradict UHH2 code?
            alpha_raw,  # 1,
        ),
    )

    # set output column
    events = set_ak_column(events, self.alpha_name, alpha)

    return events


@alpha.init
def alpha_init(self: Producer) -> None:
    self.uses.add(f"{self.jet_name}.pt")
    self.produces.add(f"{self.alpha_name}")
