# coding: utf-8
"""
Lookup corresponding events in UHH2.
"""

import law

from law.util import InsertableDict

from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.normalization import normalization_weights
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.muon import muon_weights
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column, update_ak_array

from dijet.production.default import default

np = maybe_import("numpy")
ak = maybe_import("awkward")


@producer(
    uses={"run", "luminosityBlock", "event"},
    produces={"uhh2.*"},
)
def uhh2(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # indexes of events in UHH2 array
    uhh2_rows = [
        self.uhh2_lookup.get(rle)
        for rle in zip(
            events.run,
            events.luminosityBlock,
            events.event,
        )
    ]
    # UHH2 events in order they appear in CF
    cf_events = self.uhh2_events[uhh2_rows]

    # write out
    for field in cf_events.fields:
        # replace nan values
        values = ak.nan_to_num(
            cf_events[field],
            nan=EMPTY_FLOAT,
        )

        # write out events
        events = set_ak_column(
            events,
            f"uhh2.{field}",
            values,
        )

    return events


@uhh2.requires
def uhh2_requires(self: Producer, task: law.Task, reqs: dict) -> None:
    """
    Register `UHH2ToParquet` task as dependency.
    """
    if "uhh2" in reqs:
        return

    from dijet.tasks.uhh2 import UHH2ToParquet
    reqs["uhh2"] = UHH2ToParquet.req(task)


@uhh2.setup
def uhh2_setup(
    self: Producer,
    task: law.Task,
    reqs: dict,
    inputs: dict,
    reader_targets: InsertableDict,
) -> None:
    """
    Load UHH2 events and create hash map
    """

    uhh2_fname = reqs["uhh2"].output().path
    print(f"Loading UHH2 events from file: {uhh2_fname}")
    self.uhh2_events = ak.from_parquet(uhh2_fname)

    # dictionary for lookup by (run, ls, event)
    print(f"Computing lookup table")
    self.uhh2_lookup = {
        e: i
        for i, e in enumerate(zip(
            self.uhh2_events.run,
            self.uhh2_events.lumi_sec,
            self.uhh2_events.eventID,
        ))
    }


@producer(
    uses={"run", "luminosityBlock", "event", category_ids},
    produces={category_ids},
)
def uhh2_categories(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # re-run category IDs
    events = self[category_ids](events, **kwargs)
    return events


@uhh2_categories.pre_init
def uhh2_categories_init(self: Producer) -> None:
    from dijet.config.uhh2 import add_uhh2_categories

    # add UHH2-related categories
    add_uhh2_categories(self.config_inst)


@uhh2_categories.requires
def uhh2_categories_requires(self: Producer, task: law.Task, reqs: dict) -> None:
    """
    Register `UHH2ToParquet` task as dependency.
    """
    from columnflow.tasks.production import ProduceColumns

    # can't call ProduceColumns.req directly due to buggy behavior
    # (tested with CF revision 8004828cd7f1f61600b027cff335476de95b9647)
    # 1. 'producer' is part of _prefer_cli, so whatever we pass here will be ignored in favor of the CLI option
    # 2. 'producer_inst' is not resolved unless 'known_shifts' is None
    # -> work around the issue by constructing the req_params ourselves
    req_params = ProduceColumns.req_params(task)
    req_params.pop("producer", None)
    req_params.pop("producer_inst", None)
    req_params.pop("known_shifts", None)  # need to remove for instance resolution to work -> bug?

    if "default" not in reqs:
        reqs["default"] = ProduceColumns(**req_params, producer="default")

    if "uhh2" not in reqs:
        reqs["uhh2"] = ProduceColumns(**req_params, producer="uhh2")


@uhh2_categories.setup
def uhh2_categories_setup(
    self: Producer,
    task: law.Task,
    reqs: dict,
    inputs: dict,
    reader_targets: InsertableDict,
) -> None:
    """
    Load UHH2 events and create hash map
    """
    reader_targets["default"] = inputs["default"]["columns"]
    reader_targets["uhh2"] = inputs["uhh2"]["columns"]
