# coding: utf-8

"""
Custom tasks to derive JER SF.
"""

import law

from dijet.tasks.base import HistogramsBaseTask
from columnflow.util import maybe_import
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

    # declare output collection type and keys
    output_collection_cls = law.NestedSiblingFileCollection
    output_base_keys = ("sfs",)

    # how to create the branch map
    branching_type = "merged"

    # upstream requirements
    reqs = Requirements(
        JER=JER,
    )

    #
    # methods required by law
    #

    def output(self):
        """
        Organize output as a (nested) dictionary. Output files will be in a single
        directory, which is determined by `store_parts`.
        """
        return {
            key: self.target(f"{key}.pickle")
            for key in self.output_base_keys
        }

    def requires(self):
        return self.reqs.JER.req_different_branching(self, branch=-1)

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["key"] = self.requires_from_branch()
        return reqs

    #
    # helper methods for handling task inputs/outputs
    #

    def load_input(self, key: str, sample: str):
        coll_keys = [
            coll_key
            for coll_key, coll in self.input()["collection"].targets.items()
            if sample in coll[key]
        ]
        if len(coll_keys) != 1:
            raise RuntimeError(
                f"found {len(coll_keys)} input collections corresponding to "
                f"sample '{sample}', expected 1",
            )
        return self.input()["collection"][coll_keys[0]][key][sample].load(formatter="pickle")

    def dump_output(self, key: str, obj: object):
        if key not in self.output_base_keys:
            raise ValueError(
                f"output key '{key}' not registered in "
                f"`{self.task_family}.output_base_keys`",
            )
        self.output()[key].dump(obj, formatter="pickle")

    #
    # task implementation
    #

    def run(self):
        print(
            f"computing SF for samples {self.branch_data.mc_sample!r} (MC), "
            f"{self.branch_data.data_sample!r} (data)",
        )

        # load JERs results
        jers = {
            "data": self.load_input("jers", sample=self.branch_data.data_sample)["jer"],
            "mc": self.load_input("jers", sample=self.branch_data.mc_sample)["jer"],
        }

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
