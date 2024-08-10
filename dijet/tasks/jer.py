# coding: utf-8

"""
Custom tasks to derive JER SF.
"""

import law

from columnflow.util import maybe_import
from columnflow.tasks.framework.base import Requirements

from dijet.tasks.base import HistogramsBaseTask
from dijet.tasks.alpha import AlphaExtrapolation

hist = maybe_import("hist")
np = maybe_import("numpy")


class JER(
    HistogramsBaseTask,
):
    """
    Task to calculate JER after alpha extrapolation.

    Processing steps:
    - read in extrapolated widths from `AlphaExtrapolation` task
    - subtract gen-level widths (PLI) from reco (is data use gen-width from MC)
    - calculate JER in using standard method (SM) and forward-extension (FE) methods
    """

    # declare output as a nested sibling file collection
    output_collection_cls = law.NestedSiblingFileCollection
    output_base_keys = ("jers",)
    output_per_level = False

    subtract_pli = law.OptionalBoolParameter(
        description="if True, will subtract the gen-level asymmetry (a.k.a. particle-level "
        "imbalance or PLI) from the extrapolated widths before deriving JER",
        default=True,
    )

    # upstream requirements
    reqs = Requirements(
        AlphaExtrapolation=AlphaExtrapolation,
    )

    def requires(self):
        deps = {}

        # require extrapolation results
        deps["reco"] = self.reqs.AlphaExtrapolation.req(
            self,
            levels="reco",
        )

        # if PLI subtraction requested, also require
        # gen-level extrapolation results
        if self.subtract_pli:
            deps["gen"] = self.reqs.AlphaExtrapolation.req(
                self,
                levels="gen",
                # set single process by hand and get first
                # branch to ensure gen-level information is
                # read for MC even when running on data
                # TODO: kinda hacky, improve
                processes=("qcd",),
                branch=0,  # branches=[0] (?)
            )

        return deps

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["key"] = self.requires_from_branch()
        return reqs

    def load_extrapolation(self, level: str):
        key = self._io_key("extrapolation", level=level)
        histogram = self.input()[level][key].load(formatter="pickle")
        return histogram

    def run(self):
        # load extrapolation results
        results_extrapolation = self.load_extrapolation(level="reco")

        # get extrapolated distribution widths
        h_widths = results_extrapolation["intercepts"]

        # subtract PLI if requested
        if self.subtract_pli:
            # getrieve gen-level results
            results_extrapolation_gen = self.load_extrapolation(level="gen")
            h_widths_gen = results_extrapolation_gen["intercepts"]

            # subtract the gen-level results from the extrapolated widths
            values = np.sqrt(np.maximum(
                h_widths.values()**2 - h_widths_gen.values()**2,
                0.0,
            ))
            # Gaussian error propagation
            variances = np.nan_to_num(
                (
                    h_widths.variances() * h_widths.values()**2 +
                    h_widths_gen.variances() * h_widths_gen.values()**2
                ) / values,
                nan=0.0,
            )

            # save subtracted values back in h_widths
            v_widths = h_widths.view()
            v_widths.value = values
            v_widths.variance = variances

        # get index on `category` axis corresponding to
        # the two computation methods
        categories = list(h_widths.axes["category"])
        index_methods = {
            m: categories.index(self.LOOKUP_CATEGORY_ID[m])
            for m in self.LOOKUP_CATEGORY_ID
        }

        # calcuate JER for standard method
        jer_sm_val = h_widths[index_methods["sm"], :, :].values() * np.sqrt(2)
        jer_sm_err = np.sqrt(h_widths[index_methods["sm"], :, :].variances()) * np.sqrt(2)

        # average over first few eta bins to get
        # reference JER for forward method
        # TODO: Define eta bin in config
        jer_ref_val = np.mean(jer_sm_val[:5, :], axis=0, keepdims=True)
        jer_ref_err = np.mean(jer_sm_err[:5, :], axis=0, keepdims=True)

        # calculate JER for forward extension method
        # TODO: Check if factor 2 or 4. Keep consistent with UHH2 for now
        jer_fe_val = np.sqrt(2 * h_widths[index_methods["fe"], :, :].values()**2 - jer_ref_val**2)
        term_probe = 2 * h_widths[index_methods["fe"], :, :].values() * h_widths[index_methods["fe"], :, :].variances()
        term_ref = jer_ref_val * jer_ref_err
        jer_fe_err = np.sqrt(term_probe**2 + term_ref**2) / jer_fe_val

        # create output histogram and view for filling
        h_jer = h_widths.copy()
        v_jer = h_jer.view()

        # write JER values to output histogram
        v_jer[index_methods["sm"], :, :].value = np.nan_to_num(jer_sm_val, nan=0.0)
        v_jer[index_methods["sm"], :, :].variance = np.nan_to_num(jer_sm_err**2, nan=0.0)
        v_jer[index_methods["fe"], :, :].value = np.nan_to_num(jer_fe_val, nan=0.0)
        v_jer[index_methods["fe"], :, :].variance = np.nan_to_num(jer_fe_err**2, nan=0.0)

        results_jers = {
            "jer": h_jer,
        }
        self.output()["jers"].dump(results_jers, formatter="pickle")
