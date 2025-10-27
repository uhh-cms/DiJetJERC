# coding: utf-8

import order as od

from functools import wraps

from columnflow.util import maybe_import

ak = maybe_import("awkward")


def masked_sorted_indices(mask: ak.Array, sort_var: ak.Array, ascending: bool = False) -> ak.Array:
    """
    Helper function to obtain the correct indices of an object mask
    """
    indices = ak.argsort(sort_var, axis=-1, ascending=ascending)
    return indices[mask[indices]]


def call_once_on_config(func=None, *, include_hash=False):
    """
    Parametrized decorator to ensure that function *func* is only called once for the config *config*.
    Can be used with or without parentheses.
    """
    if func is None:
        # If func is None, it means the decorator was called with arguments.
        def wrapper(f):
            return call_once_on_config(f, include_hash=include_hash)
        return wrapper

    @wraps(func)
    def inner(config, *args, **kwargs):
        tag = f"{func.__name__}_called"
        if include_hash:
            tag += f"_{func.__hash__()}"

        if config.has_tag(tag):
            return

        config.add_tag(tag)
        return func(config, *args, **kwargs)

    return inner


def get_variable_for_level(config: od.Config, name: str, level: str):
    """
    Get name of variable corresponding to main variable *name*
    of level *level*, where *level* can be either 'reco' or 'gen'.
    """
    if level == "reco":
        # reco level is default -> return directly
        return name
    elif level == "gen":
        # look up registered gen-level name in aux data
        var_inst = config.get_variable(name)
        return var_inst.x("gen_variable", name)
    else:
        raise ValueError(f"invalid level '{level}', expected one of: gen,reco")
