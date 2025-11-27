# coding: utf-8
from __future__ import annotations

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


def get_nested_entry(input_dict: dict, key: tuple[str] | str, on_missing: str = "raise"):
    # validate inputs
    if on_missing not in ("raise", "return_none"):
        raise ValueError(
            f"invalid value {on_missing=!r}, expecting one of: raise,return_none",
        )

    # split string key by "." to get consecutive fields to check
    if isinstance(key, str):
        key = key.split(".")

    # check presence of nested fields
    val = input_dict
    for i_key, key_ in enumerate(key):
        # return early if not found
        if (val := val.get(key_, None)) is None:
            if on_missing == "raise":
                missing_key_str = ".".join(key[:i_key + 1])
                raise KeyError(missing_key_str)
            elif on_missing == "return_none":
                return None

    # return value
    return val


def deflate_dict(input_dict: dict, sep: str = ".", max_depth: int | None = None):
    """
    Deflate a nested dictionary to one containing a single level of keys.

    The returned dictionary will have keys of the form ``level_1.level_2.[...].level_N``,
    where ``level_i`` refer to the keys applied at the *i*-th level, e.g.:
    ```
    flat_dict["level_1.level_2"] == input_dict["level_1"]["level_2"]
    ```

    If given, *max_depth* indicates the maximum number of key levels to merge, starting
    from the top level. A *max_depth* of 1 will leave the input dict unchanged.
    """
    flat_dict = {}

    def _flatten(structured_dict: dict, prev_keys: list[str] | None = None):
        prev_keys = prev_keys or []
        depth = len(prev_keys)
        for key, value in structured_dict.items():
            if isinstance(value, dict) and (max_depth is None or depth < max_depth - 1):
                _flatten(value, prev_keys + [key])
                continue

            flat_key = sep.join(prev_keys + [key])
            flat_dict[flat_key] = value

    _flatten(input_dict)

    return flat_dict


def inflate_dict(flat_dict: dict):
    """
    Inflate a single-level dictionary with keys of the form 'level_1.level_2.[...]'.

    The returned dict will have a nested structure with keys indicated by
    the 'level_i' substrings, e.g.:
    ```
    input_dict["level_1"]["level_2"] == flat_dict["level_1.level_2"]
    ```
    """
    nested_dict = {}

    for key, value in flat_dict.items():
        key_seq = key.split(".")

        inner_dict = nested_dict
        for key_ in key_seq[:-1]:
            if key_ not in inner_dict:
                inner_dict[key_] = {}

            inner_dict = inner_dict[key_]

        if key_seq[-1] in inner_dict:
            raise ValueError(
                f"direct value provided for nested key '{key}'",
            )

        inner_dict[key_seq[-1]] = value

    return nested_dict


# helper function for iterating through a nested
# dict in a flat way
def iter_flat_dict(d: dict):
    """
    Iterate through a nested dictionary in a flat way.
    """
    if not isinstance(d, dict):
        yield d
        return
    for k, v in d.items():
        yield from iter_flat_dict(v)
