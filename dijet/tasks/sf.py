# coding: utf-8

"""
Custom tasks to derive JER SF.
"""

import law

# from dijet.tasks.base import HistogramsBaseTask
from columnflow.util import maybe_import
from columnflow.tasks.framework.base import Requirements

from dijet.tasks.jer import JER
from dijet.tasks.merge import DataMCTaskBase

np = maybe_import("numpy")


class SF(DataMCTaskBase):
    """
    Task to derive the JER SFs and store those in a dedicated file.
    The input for this task is taken from the JER output from data and MC.
    The JER SFs are derived from the ratio of the JER from data and MC.
    TODO: - Add NSC fit
          - Add constant fit
    """

    output_collection_cls = law.NestedSiblingFileCollection

    # upstream requirements
    reqs = Requirements(
        JER=JER,
    )

    def requires(self):
        return self.reqs.JER.req(
            self,
            processes=("qcd", "data"),
        )

    def load_jers(self):
        return (
            self.input().collection[0]["jers"].load(formatter="pickle")["jer"],
            self.input().collection[1]["jers"].load(formatter="pickle")["jer"],
        )

    def output(self) -> dict[law.FileSystemTarget]:
        # TODO: Into base and add argument alphas, jers, etc.
        sample = self.extract_sample()
        target = self.target(f"{sample}", dir=True)
        outp = {
            "sfs": target.child("sfs.pickle", type="f"),
        }
        return outp

    def run(self):
        # load JERs results
        jers_da, jers_mc = self.load_jers()
        # from IPython import embed;embed()

        v_jers_da = jers_da.view().copy()
        v_jers_mc = jers_mc.view().copy()

        h_sf = jers_da.copy()
        v_sf = h_sf.view()

        v_sf.value = v_jers_da.value / v_jers_mc.value
        # inf values if mc is zero; add for precaution
        mask = np.fabs(v_sf.value) == np.inf  # account also for -inf
        v_sf.value[mask] = np.nan
        v_sf.value = np.nan_to_num(v_sf.value, nan=0.0)

        # Error propagation
        # x = data; y = mc; s_x = sigma x
        # x/y -> sqrt( ( s_x/y )**2 + ( (x*s_y)/y**2 )**2 )
        term1 = v_jers_da.variance / v_jers_mc.value
        term2 = (v_jers_da.value * v_jers_mc.variance) / v_jers_mc.value**2
        v_sf.variance = np.sqrt(term1**2 + term2**2)
        # inf values if mc is zero; add for precaution
        v_sf.variance[mask] = np.nan  # account also for -inf
        v_sf.variance = np.nan_to_num(v_sf.variance, nan=0.0)

        results_sfs = {
            "sfs": h_sf,
        }
        self.output()["sfs"].dump(results_sfs, formatter="pickle")
