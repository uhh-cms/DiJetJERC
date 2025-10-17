# coding: utf-8
"""
UHH2 synchronization config related code
"""

import law

from columnflow.util import maybe_import
from columnflow.columnar_util import Route
# from columnflow.config_util import create_category_combinations

from columnflow.categorization import Categorizer, categorizer
from dijet.production.jet_assignment import jet_assignment
from dijet.production.uhh2 import uhh2

from dijet.util import call_once_on_config

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
    return False  # don't skip


@call_once_on_config()
def add_uhh2_categories(config: od.Config) -> None:
    """
    Adds UHH2 synchronization related categories to a *config*.
    """

    #
    # group 2: UHH2 synchronization related categories 
    #

    cat_idx_lsd = 1
    uhh2_synch_categories = []

    @categorizer(
        uses={
            "event",
            "probe_jet.{pt,eta,phi}",
            "uhh2.probejet_{pt,eta,phi}",
        },
        cls_name="catid_uhh2_same_probe_jet",
    )
    def catid_uhh2_same_probe_jet(
        self: Categorizer, events: ak.Array,
        **kwargs,
    ) -> ak.Array:
        """
        Select events where UHH2 and CF randomly selected the same probe jet.
        """
        mask = ak.ones_like(events.event) > 0

        # compare fields with tolerance
        for var_name, var_tol in [
            ("pt", 1.0),
            ("eta", 1e-3),
            ("phi", 1e-3),
        ]:
            val_cf = Route(f"probe_jet.{var_name}").apply(events, None)
            val_uhh2 = Route(f"uhh2.probejet_{var_name}").apply(events, None)
            vals_equal_tolerant = abs(val_cf - val_uhh2) < var_tol
            mask = mask & vals_equal_tolerant

        return events, ak.fill_none(mask, False)

    uhh2_synch_categories.append(
        config.add_category(
            name="uhh2_same_probe_jet",
            id=int(10**cat_idx_lsd),
            selection="catid_uhh2_same_probe_jet",
            label="UHH2/CF same probe jet",
        ),
    )
