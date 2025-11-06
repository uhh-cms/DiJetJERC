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


def product_dict(input_dict: dict):
    """
    Generator that yields the Cartesian product of iterables in a dictionary.
    Yielded elements are dictionaries that map keys of the original dictionary
    to individual elements of the input sequences
    """
    from itertools import product

    keys = input_dict.keys()
    for instance in product(*input_dict.values()):
        yield dict(zip(keys, instance))


def iter_bins(bin_edges, **add_kwargs):
    """
    Generator that takes a sequence of (ascending) bin edges and yields a sequence
    of dicts corresponding to individual bins. Dicts contain the following keys:
    - *index*, the index of the current bin, starting from 0;
    - *lo*, the lower edge of the bin;
    - *hi*, the upper edge of the bin;
    - any additional key-value pairs provided via *add_kwargs*.
    """
    # check reserved kwargs and raise
    reserved_keys = set(add_kwargs).intersection({"index", "up", "dn"})
    if reserved_keys:
        reserved_keys_str = ",".join(sorted(reserved_keys))
        raise ValueError(
            "the following keys are reserved and cannot be present in add_kwargs: "
            f"{reserved_keys_str}",
        )

    # yield bin dict
    for i, (e_lo, e_hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        yield {"index": i, "lo": e_lo, "hi": e_hi, **add_kwargs}
