# coding: utf-8

"""
Definition of categories.
"""

from collections import OrderedDict

import law

from columnflow.config_util import create_category_combinations

import order as od

logger = law.logger.get_logger(__name__)


def add_categories_selection(config: od.Config) -> None:
    """
    Adds categories to a *config*, that are typically produced in `SelectEvents`.
    """
    config.add_category(
        name="incl",
        id=1,
        selection="catid_selection_incl",
        label="Inclusive",
    )

def name_fn(**root_cats):
    cat_name = "__".join(cat for cat in root_cats.values())
    return cat_name


def kwargs_fn(root_cats):
    kwargs = {
        "id": sum([c.id for c in root_cats.values()]),
        "label": ", ".join([c.name for c in root_cats.values()]),
    }
    return kwargs

