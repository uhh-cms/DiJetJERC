# coding: utf-8

"""
Custom tasks to derive JER SF.
"""

import law

from dijet.tasks.base import HistogramsBaseTask
from columnflow.util import maybe_import, DotDict
from columnflow.tasks.framework.base import Requirements

from dijet.tasks.alpha import AlphaExtrapolation

ak = maybe_import("awkward")
hist = maybe_import("hist")
np = maybe_import("numpy")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")
it = maybe_import("itertools")


class JER(HistogramsBaseTask):
    """
    Task to perform alpha extrapolation.
    Read in and plot asymmetry histograms.
    Cut of non-gaussian tails and extrapolate sigma_A( alpha->0 ).
    """

    output_collection_cls = law.NestedSiblingFileCollection

    # upstream requirements
    reqs = Requirements(
        AlphaExtrapolation=AlphaExtrapolation,
    )

    def create_branch_map(self):
        return [
            DotDict({"process": process})
            for process in sorted(self.processes)
        ]

    def requires(self):
        return self.reqs.AlphaExtrapolation.req(
            self,
            branch=-1,
            _exclude={"branches"},
        )

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["key"] = self.as_branch().requires()
        return reqs

    def load_alpha(self):
        histogram = self.input().collection[0]["alphas"].load(formatter="pickle")
        return histogram

    def output(self) -> dict[law.FileSystemTarget]:
        # TODO: Into base and add argument alphas, jers, etc.
        sample = self.extract_sample()
        target = self.target(f"{sample}", dir=True)
        outp = {
            "jers": target.child("jers.pickle", type="f"),
        }
        return outp

    def run(self):
        widths = self.load_alpha()

        # ### Now JER SM Data

        jer_sm = widths["sm"]["fits"][0][:, :] * np.sqrt(2)

        # TODO: Define eta bin in config
        # NOTE: weighting number by appearence in eta regions
        jer_ref = np.mean(jer_sm[:5, :], axis=0).reshape(1, -1)
        jer_fe = np.sqrt(4 * widths["fe"]["fits"][0][:, :]**2 - jer_ref**2)

        results_jers = {
            "sm": {
                "jers": jer_sm,   
            },
            "fe": {
                "jers": jer_fe,  
            },
        }
        self.output()["jers"].dump(results_jers, formatter="pickle")
