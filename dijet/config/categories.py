# coding: utf-8

"""
Definition of categories.

Categories are assigned a unique integer ID according to a fixed numbering
scheme, with digits/groups of digits indicating the different category groups:

    lowest digit
      |
    +---+
    | M |
    +---+

+=======+===============+======================+=================================+
| Digit | Description   | Values               | Category name                   |
+=======+===============+======================+=================================+
| M     | Method        |                      | Choose Method                   |
|       |               | 1: SM                |  - Standard Method              |
|       |               |                      |    jet1 and jet2 same eta bin   |
|       |               | 2: FE                |  - Forward Extension            |
|       |               |                      |    jet1 < 1.131 or              |
|       |               |                      |    jet2 < 1.131                 |
|       |               |                      |    jet in barrel: reference jet |
|       |               |                      |    jet not in barrel: probe jet |
|       |               |                      |    If both in barrel, they need |
|       |               |                      |    to be in different eta bins  |
|       |               |                      |    (SM flag must be false)      |
|       |               |                      |    JER measured from probe jet  |
| x 1   |               |                      |    JER from FE dependent on SM  |
+-------+---------------+----------------------+---------------------------------+

A digit group consisting entirely of zeroes ('0') represents the inclusive
category for the corresponding category group, i.e. no selection from that
group is applied.

This scheme encodes child/parent relations into the ID, making it easy
to check if categories overlap or are subcategories of each other. When applied
to a set of categories from different groups, the sum of the category IDs is the
ID of the combined category.
"""

import itertools
import law

from columnflow.util import maybe_import
from columnflow.config_util import create_category_combinations
from columnflow.categorization import Categorizer, categorizer

from dijet.production.dijet_balance import dijet_balance
from dijet.production.jet_assignment import jet_assignment

import order as od

logger = law.logger.get_logger(__name__)

np = maybe_import("numpy")
ak = maybe_import("awkward")


def name_fn(categories: dict[str, od.Category]):
    """Naming function for automatically generated combined categories."""
    return "__".join(cat.name for cat in categories.values() if cat)


def kwargs_fn(categories: dict[str, od.Category]):
    """Customization function for automatically generated combined categories."""
    return {
        "id": sum(cat.id for cat in categories.values()),
        "selection": [cat.selection for cat in categories.values()],
        "label": "\n".join(
            cat.label for cat in categories.values()
        ),
    }


def skip_fn(categories: dict[str, od.Category]):
    """Custom function for skipping certain category combinations."""
    # skip if combination involves both `alpha` and `alpha_incl` groups,
    # since these are not disjoint

    return False  # don't skip


def add_categories(config: od.Config) -> None:
    """
    Adds categories to a *config*, that are typically produced in `SelectEvents`.
    """
    # inclusive category
    config.add_category(
        name="incl",
        id=0,
        selection="sel_incl",
        label="inclusive",
    )

    #
    # group 1: Standard Method or forward extension
    #

    cat_idx_lsd = 0
    cat_idx_ndigits = 1
    method_categories = []

    @categorizer(
        uses={jet_assignment},
        cls_name="sel_sm",
    )
    def sel_sm(
        self: Categorizer, events: ak.Array,
        **kwargs,
    ) -> ak.Array:
        """
        Select events with probe jet and reference jet in same eta bin (standard method)
        """
        events = self[jet_assignment](events, **kwargs)
        return events, ak.fill_none(events.use_sm, False)

    method_categories.append(
        config.add_category(
            name="sm",
            id=int(10**cat_idx_lsd),
            selection="sel_sm",
            label="sm",
        ),
    )

    @categorizer(
        uses={jet_assignment},
        cls_name="sel_fe",
    )
    def sel_fe(
        self: Categorizer, events: ak.Array,
        **kwargs,
    ) -> ak.Array:
        """
        Select events with
        probe jet eta > 1.131 and
        reference jet eta < 1.131
        (forward extension)
        """
        events = self[jet_assignment](events, **kwargs)
        return events, ak.fill_none(events.use_fe, False)

    cat = config.add_category(
        name="sel_fe",
        id=int(10**cat_idx_lsd * 2),
        selection="sel_fe",
        label="fe",
    )
    method_categories.append(cat)
