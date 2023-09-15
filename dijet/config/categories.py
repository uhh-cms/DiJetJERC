# coding: utf-8

"""
Definition of categories.

Categories are assigned a unique integer ID according to a fixed numbering
scheme, with digits/groups of digits indicating the different category groups:

               lowest digit
                          |
    +---+---+---+---+---+---+
    | I | A | P | P | E | E |
    +---+---+---+---+---+---+

+=======+===============+======================+=================================+
| Digit | Description   | Values               | Category name                   |
+=======+===============+======================+=================================+
| E     | eta bins      | 1 to e               | eta_{eta_min}_{eta_max}         |
|    10 |               |                      |                                 |
+-------+---------------+----------------------+---------------------------------+
| P     | pt avg bin    | 1 to p               | pt_{pt_min}_{pt_max}            |
|  1000 |               |                      |                                 |
+-------+---------------+----------------------+---------------------------------+
| A     | alpha binning | 1 to a               | alpha_{min}_{max} (exclusive)   |
| x 1e4 |               |                      | or                              |
+-------+---------------+----------------------+ alpha_lt_{max} (inclusive)      |
| I     | Inclusive     |                      |                                 |
|       | alpha flag    |                      | In combination with alpha bins  |
|       |               | 0: exclusive         | Excl: alpha in one bin          |
|       |               | 1: inclusive         | Incl: All alpha below the upper |
|       |               |                      |       bin value                 |
|       |               |                      | e.g.  bin IA=06 -> 0.25<a<0.3   |
| x 1e5 |               |                      |       bin IA=16 ->      a<0.3   |
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
from dijet.constants import pt, eta, alpha

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
    if all(group in categories for group in ["alpha", "alpha_incl"]):
        return True

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
    # group 1: probe jet eta bins
    #

    cat_idx_lsd = 0  # 10-power of least significant digit
    cat_idx_ndigits = 2  # number of digits to use for category group

    # get pt bins from config
    # TODO: Add binning in config ?
    eta_bins = eta
    eta_categories = []

    for cat_idx, (eta_min, eta_max) in enumerate(
        zip(eta_bins[:-1], eta_bins[1:]),
    ):
        eta_min_repr = f"{eta_min}".replace(".", "p")
        eta_max_repr = f"{eta_max}".replace(".", "p")
        cat_label = rf"{eta_min} $\leq$ $\left|\eta\right|$ < {eta_max}"

        cat_name = f"eta_{eta_min_repr}_{eta_max_repr}"
        sel_name = f"sel_{cat_name}"

        @categorizer(
            uses={jet_assignment},
            cls_name=sel_name,
        )
        def sel_eta(
            self: Categorizer, events: ak.Array,
            eta_range: tuple = (eta_min, eta_max),
            eta_min_repr=eta_min_repr,
            eta_max_repr=eta_max_repr,
            **kwargs,
        ) -> ak.Array:
            f"""
            Select events with probe jet eta the range [{eta_min_repr}, {eta_max_repr})
            """
            events = self[jet_assignment](events, **kwargs)
            return events, ak.fill_none(
                (events.probe_jet.eta >= eta_range[0]) &
                (events.probe_jet.eta < eta_range[1]),
                False,
            )

        assert cat_idx < 10**cat_idx_ndigits - 1, "no space for category, ID reassignement necessary"
        cat = config.add_category(
            name=cat_name,
            id=int(10**cat_idx_lsd * (cat_idx + 1)),
            selection=sel_name,
            label=cat_label,
        )

        eta_categories.append(cat)

    #
    # group 2: pt avg bins
    #
    cat_idx_lsd += cat_idx_ndigits
    cat_idx_ndigits = 2

    # get pt bins from config
    pt_bins = pt
    pt_categories = []

    for cat_idx, (pt_min, pt_max) in enumerate(
        zip(pt_bins[:-1], pt_bins[1:]),
    ):
        pt_min_repr = f"{int(pt_min)}"
        pt_max_repr = f"{int(pt_max)}"
        cat_label = rf"{pt_min} $\leq$ $p_{{T,avg}}$ < {pt_max} GeV"

        cat_name = f"pt_{pt_min_repr}_{pt_max_repr}"
        sel_name = f"sel_{cat_name}"

        @categorizer(
            uses={dijet_balance},
            cls_name=sel_name,
        )
        def sel_pt(
            self: Categorizer, events: ak.Array,
            pt_range: tuple = (pt_min, pt_max),
            pt_min_repr=pt_min_repr,
            pt_max_repr=pt_max_repr,
            **kwargs,
        ) -> ak.Array:
            f"""
            Select events with probe jet pt the range [{pt_min_repr}, {pt_max_repr})
            """
            events = self[dijet_balance](events, **kwargs)
            return events, ak.fill_none(
                (events.dijets.pt_avg >= pt_range[0]) &
                (events.dijets.pt_avg < pt_range[1]),
                False,
            )
        assert cat_idx < 10**cat_idx_ndigits - 1, "no space for category, ID reassignement necessary"
        cat = config.add_category(
            name=cat_name,
            id=int(10**cat_idx_lsd * (cat_idx + 1)),
            selection=sel_name,
            label=cat_label,
        )

        pt_categories.append(cat)

    #
    # group 3: alpha bins
    #

    cat_idx_lsd += cat_idx_ndigits  # 10-power of least significant digit
    cat_idx_ndigits = 1  # number of digits to use for category group

    # get pt bins from config
    alpha_bins = alpha  # TODO: Add binning in config ?
    alpha_categories = []

    for cat_idx, (alpha_min, alpha_max) in enumerate(
        zip(alpha_bins[:-1], alpha_bins[1:]),
    ):
        alpha_min_repr = f"{alpha_min}".replace(".", "p")
        alpha_max_repr = f"{alpha_max}".replace(".", "p")
        cat_label = rf"{alpha_min} $\leq$ $\alpha$ < {alpha_max}"

        cat_name = f"alpha_{alpha_min_repr}_{alpha_max_repr}"
        sel_name = f"sel_{cat_name}"

        @categorizer(
            uses={dijet_balance},
            cls_name=sel_name,
        )
        def sel_alpha(
            self: Categorizer, events: ak.Array,
            alpha_range: tuple = (alpha_min, alpha_max),
            alpha_min_repr=alpha_min_repr,
            alpha_max_repr=alpha_max_repr,
            **kwargs,
        ) -> ak.Array:
            f"""
            Select events with probe jet alpha the range [{alpha_min_repr}, {alpha_max_repr})
            """
            events = self[dijet_balance](events, **kwargs)
            return events, ak.fill_none(
                (events.dijets.alpha >= alpha_range[0]) &
                (events.dijets.alpha < alpha_range[1]),
                False,
            )

        assert cat_idx < 10**cat_idx_ndigits - 1, "no space for category, ID reassignement necessary"
        cat = config.add_category(
            name=cat_name,
            id=int(10**cat_idx_lsd * (cat_idx + 1)),
            selection=sel_name,
            label=cat_label,
        )

        alpha_categories.append(cat)

    # inclusive alpha categories from union of (exclusive) alpha bins
    cat_idx_ndigits = 2
    alpha_incl_bins = [
        f"lt_{str(a).replace('.', 'p')}" for a in alpha[1:]
    ]
    assert len(alpha_incl_bins) + 1 == len(alpha)
    alpha_incl_categories = []
    for cat_idx, (alpha_bin, alpha_val) in enumerate(zip(alpha_incl_bins, alpha_bins[1:])):
        cat_label = rf"$\alpha$ < {alpha_val}"

        cat_name = f"alpha_{alpha_bin}"
        sel_name = f"sel_{cat_name}"

        # create category and add individual alpha intervals as child categories
        cat = config.add_category(
            name=cat_name,
            id=int(10**cat_idx_lsd * ((cat_idx) + 10**cat_idx_lsd + 1)),
            selection=None,
            label=cat_label,
        )
        child_cats = alpha_categories[:cat_idx + 1]
        for child_cat in child_cats:
            cat.add_category(child_cat)

        alpha_incl_categories.append(cat)

    # -- combined categories

    def add_combined_categories(config):
        if getattr(config, "has_combined_categories", False):
            return  # combined categories already added

        category_groups = {
            "eta": eta_categories,
            "pt": pt_categories,
            "alpha": alpha_categories,
            "alpha_incl": alpha_incl_categories,
        }

        create_category_combinations(
            config,
            category_groups,
            name_fn,
            kwargs_fn,
            skip_fn=skip_fn,
            skip_existing=False,
        )

        # connect intermediary `alpha_incl` and `alpha` categories
        category_groups_no_alpha = {
            "eta": eta_categories,
            "pt": pt_categories,
        }
        # go through all possible combinations of category *groups*
        for n in range(1, len(category_groups_no_alpha) + 1):
            for group_names in itertools.combinations(category_groups_no_alpha, n):
                # get root categories for each group
                root_cats = [category_groups_no_alpha[gn] for gn in group_names]
                # go through cross product of *root categories* (one per group)
                for root_cats in itertools.product(*root_cats):
                    root_cats = dict(zip(group_names, root_cats))
                    # go through non-compound inclusive alpha
                    # categories and their children
                    for alpha_incl_cat in alpha_incl_categories:
                        root_cats_1 = dict(root_cats, **{"alpha_incl": alpha_incl_cat})
                        name_1 = name_fn(root_cats_1)
                        cat_1 = config.get_category(name_1)
                        for alpha_cat in alpha_incl_cat.categories:
                            # skip compound children
                            if "__" in alpha_cat.name:
                                continue
                            root_cats_2 = dict(root_cats, **{"alpha": alpha_cat})
                            name_2 = name_fn(root_cats_2)
                            cat_2 = config.get_category(name_2)
                            # add parent-child relationship between compound
                            # alpha inclusive and exclusive categories
                            cat_1.add_category(cat_2)

        config.has_combined_categories = True

    add_combined_categories(config)
