# coding: utf-8

"""
Custom tasks to derive JER SF.
"""

import law

from dijet.tasks.base import HistogramsBaseTask
from columnflow.util import maybe_import, DotDict
from columnflow.tasks.framework.base import Requirements

from dijet.tasks.jer import JER

np = maybe_import("numpy")


class SF(
    HistogramsBaseTask,
    law.LocalWorkflow,
):
    """
    Task to derive the JER SFs from the ratio of JERs in data and MC.

    Processing steps:
    - read in JERs for both data and MC from `JER` task
    - perform a ratio
    - TODO: add fits (constant, NSC)
    """

    # declare output as a nested sibling file collection
    output_collection_cls = law.NestedSiblingFileCollection
    output_base_keys = ("sfs",)
    output_per_level = False

    # how to create the branch map
    branching_type = "merged"

    # upstream requirements
    reqs = Requirements(
        JER=JER,
    )

    def create_branch_map(self):
        """
        Workflow has exactly one branch, corresponding to the 'data' process
        for which scale factors should be computed.
        """
        branch_map = super().create_branch_map()
        # TODO: don't hardcode data sample name
        return [b for b in branch_map if b.sample == "data"]

    def requires(self):
        # require JERs for both data and MC
        # TODO: don't hardcode sample names
        return {
            "data": self.reqs.JER.req_different_branching(
                self,
                samples=("data",),
            ),
            "mc": self.reqs.JER.req_different_branching(
                self,
                samples=("qcdht",),
            ),
        }

    def load_jers(self):
        """
        Load JERs from inputs in data and MC
        """
        return {
            key: self.input()[key]["jers"].load(formatter="pickle")["jer"]  # noqa
            for key in ("data", "mc")
        }

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["key"] = self.requires_from_branch()
        return reqs


    def run(self):
        # load JERs results
        jers = self.load_jers()

        # views
        v_jers = {}
        v_jers["data"] = jers["data"].view().copy()
        v_jers["mc"] = jers["mc"].view().copy()

        # output histogram with scale factors
        h_sf = jers["data"].copy()
        v_sf = h_sf.view()

        v_sf.value = v_jers["data"].value / v_jers["mc"].value
        # inf values if mc is zero; add for precaution
        mask = np.fabs(v_sf.value) == np.inf  # account also for -inf
        v_sf.value[mask] = np.nan
        v_sf.value = np.nan_to_num(v_sf.value, nan=0.0)

        # Error propagation
        # x = data; y = mc; s_x = sigma x
        # x/y -> sqrt( ( s_x/y )**2 + ( (x*s_y)/y**2 )**2 )
        term1 = v_jers["data"].variance / v_jers["mc"].value
        term2 = (v_jers["data"].value * v_jers["mc"].variance) / v_jers["mc"].value**2
        v_sf.variance = np.sqrt(term1**2 + term2**2)
        # inf values if mc is zero; add for precaution
        v_sf.variance[mask] = np.nan  # account also for -inf
        v_sf.variance = np.nan_to_num(v_sf.variance, nan=0.0)

        results_sfs = {
            "sfs": h_sf,
        }
        self.output()["sfs"].dump(results_sfs, formatter="pickle")
