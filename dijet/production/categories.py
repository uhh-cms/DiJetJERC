# coding: utf-8

"""
Column production methods related defining categories.

NOTE: essentially identical to the implementation in columnflow,
but with some workarounds for analysis-specific edge cases
"""

from __future__ import annotations

from collections import defaultdict

import law

from columnflow.categorization import Categorizer
from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column


np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


@producer(produces={"category_ids"})
def category_ids(
    self: Producer,
    events: ak.Array,
    target_events: ak.Array | None = None,
    **kwargs,
) -> ak.Array:
    """
    Assigns each event an array of category ids.
    """
    category_ids = []

    for cat_inst in self.config_inst.get_leaf_categories():
        # start with a true mask
        cat_mask = np.ones(len(events), dtype=bool)

        # loop through selectors
        for categorizer in self.categorizer_map[cat_inst]:
            events, mask = self[categorizer](events, **kwargs)
            cat_mask = cat_mask & mask

        # categorizer did not select any events -> skip
        # to avoid problems on concatenating empty arrays,
        # which is still buggy in AwkwardArray
        if not ak.any(cat_mask):
            continue

        # covert to nullable array with the category ids or none
        ids = ak.mask(
            ak.ones_like(cat_mask) * cat_inst.id,
            cat_mask,
            valid_when=True,
        )

        # apply ak.singletons and pack to avoid errors on concatenate
        category_ids.append(ak.to_packed(ak.singletons(ids)))

    # combine (loop to  work wround undefined behavior bug in AwkwardArray)
    n_tries = 0
    while True:
        n_tries += 1
        try:
            category_ids = ak.concatenate(category_ids, axis=1)
        except:
            if n_tries > 20:
                print(f"Failed after {n_tries} tries!")
                raise
        else:
            break

    # save, optionally on a target events array
    if target_events is None:
        target_events = events
    target_events = set_ak_column(target_events, "category_ids", category_ids, value_type=np.int64)

    return target_events


@category_ids.init
def category_ids_init(self: Producer) -> None:
    # store a mapping from leaf category to categorizer classes for faster lookup
    self.categorizer_map = defaultdict(list)

    # add all categorizers obtained from leaf category selection expressions to the used columns
    for cat_inst in self.config_inst.get_leaf_categories():
        # treat all selections as lists of categorizers
        for sel in law.util.make_list(cat_inst.selection):
            if Categorizer.derived_by(sel):
                categorizer = sel
            elif Categorizer.has_cls(sel):
                categorizer = Categorizer.get_cls(sel)
            else:
                raise Exception(
                    f"selection '{sel}' of category '{cat_inst.name}' cannot be resolved to an "
                    "existing Categorizer object",
                )

            # the categorizer must be exposed
            if not categorizer.exposed:
                raise RuntimeError(
                    f"cannot use unexposed categorizer '{categorizer}' to evaluate category "
                    f"{cat_inst}",
                )

            # update dependency sets
            self.uses.add(categorizer)
            self.produces.add(categorizer)

            self.categorizer_map[cat_inst].append(categorizer)
