# coding: utf-8

"""
Collection of patches of underlying columnflow tasks.
"""

import os

import law
from columnflow.util import memoize


logger = law.logger.get_logger(__name__)


@memoize
def patch_bundle_repo_exclude_files():
    from columnflow.tasks.framework.remote import BundleRepo

    # get the relative path to CF_BASE
    cf_rel = os.path.relpath(os.environ["CF_BASE"], os.environ["DIJET_BASE"])

    # amend exclude files to start with the relative path to CF_BASE
    exclude_files = [os.path.join(cf_rel, path) for path in BundleRepo.exclude_files]

    # add additional files
    exclude_files.extend([
        "docs", "tests", "data", "assets", ".law", ".setups", ".data", ".github",
    ])

    # overwrite them
    BundleRepo.exclude_files[:] = exclude_files

    logger.debug("patched exclude_files of cf.BundleRepo")


# @memoize
# def patch_create_collections_from_masks():
#     """
#     Patched implementation of utility function `create_collections_from_masks`
#     to prevent the optional flag from being set on empty arrays, which an lead
#     to schema mismatches when merging job outputs.
#     """
#     from columnflow.selection import util
#
#     def create_collections_from_masks(events, object_masks):
#         """
#         Adds new collections to an *ak_array* based on *object_masks* and returns a new view.
#         *object_masks* should be a nested dictionary such as, for instance,
#
#         .. code-block:: python
#
#             {
#                 "Jet": {
#                     "BJet": ak.Array([[1, 0, 3], ...]),
#                     "LJet": ak.Array([2], ...),
#                 },
#                 ...
#             }
#
#         where outer keys refer to names of source collections and inner keys to names of collections to
#         create by applying the corresponding mask or indices to the source collection. The example above
#         would create two collections "BJet" and "LJet" based on the source collection "Jet".
#         """
#         import awkward as ak
#         from columnflow.columnar_util import set_ak_column
#
#         if isinstance(object_masks, dict):
#             object_masks = ak.Array(object_masks)
#
#         for src_name in object_masks.fields:
#             # get all destination collections
#             dst_names = list(object_masks[src_name].fields)
#
#             # when a source is named identically, handle it last
#             if src_name in dst_names:
#                 # move to the end
#                 dst_names.remove(src_name)
#                 dst_names.append(src_name)
#
#             # add collections
#             for dst_name in dst_names:
#                 object_mask = object_masks[src_name, dst_name]
#                 # bug (?) in AwkwardArray sometimes leads to an error
#                 # if object_mask has an optional type (even if no entries
#                 # are actually masked). As a workaround, remove the optional
#                 # type by filling a large index value (or False for boolean
#                 # object masks).
#                 inner_content = object_mask.layout.content.content
#                 if inner_content.is_option:
#                     dtype = inner_content.content.dtype
#                     object_mask = ak.fill_none(
#                         object_mask,
#                         False if dtype == bool else 999999,
#                     )
#                 dst_collection = events[src_name][object_mask]
#                 events = set_ak_column(events, dst_name, dst_collection)
#
#         return events
#
#     util.create_collections_from_masks = create_collections_from_masks
#     logger.debug("patched create_collections_from_masks of columnflow.selection.util")


@memoize
def patch_all():
    patch_bundle_repo_exclude_files()
    # patch_create_collections_from_masks()
