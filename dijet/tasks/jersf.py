# coding: utf-8

"""
Custom base tasks.
"""


import law

from columnflow.tasks.framework.base import Requirements, BaseTask, ShiftTask
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorStepsMixin, ProducersMixin,
    VariablesMixin, DatasetsProcessesMixin, CategoriesMixin,
)
from columnflow.tasks.reduction import MergeReducedEventsUser, MergeReducedEvents
from columnflow.tasks.production import ProduceColumns
from columnflow.tasks.histograms import MergeHistograms
from columnflow.util import dev_sandbox, maybe_import

from dijet.constants import eta, pt

ak = maybe_import("awkward")
hist = maybe_import("hist")
np = maybe_import("numpy")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")


class DiJetTask(BaseTask):

    task_namespace = "dijet"


class ColumnsBaseTask(
    DiJetTask,
    ProducersMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
    MergeReducedEventsUser,
    law.LocalWorkflow,
):
    """
    Base task to handle columns after Reduction and Production.
    An exemplary implementation of how to handle the inputs in a run method can be
    found in columnflow/tasks/union.py
    """

    exclude_index = True

    # upstream requirements
    reqs = Requirements(
        MergeReducedEventsUser.reqs,
        MergeReducedEvents=MergeReducedEvents,
        ProduceColumns=ProduceColumns,
    )

    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    def workflow_requires(self):
        reqs = super().workflow_requires()

        # require the full merge forest
        reqs["events"] = self.reqs.MergeReducedEvents.req(self, tree_index=-1)

        if not self.pilot:
            if self.producers:
                reqs["producers"] = [
                    self.reqs.ProduceColumns.req(self, producer=producer_inst.cls_name)
                    for producer_inst in self.producer_insts
                    if producer_inst.produced_columns
                ]

        return reqs

    def requires(self):
        reqs = {
            "events": self.reqs.MergeReducedEvents.req(self, tree_index=self.branch, _exclude={"branch"}),
        }

        if self.producers:
            reqs["producers"] = [
                self.reqs.ProduceColumns.req(self, producer=producer_inst.cls_name)
                for producer_inst in self.producer_insts
                if producer_inst.produced_columns
            ]

        return reqs


class HistogramsBaseTask(
    DiJetTask,
    DatasetsProcessesMixin,
    CategoriesMixin,
    VariablesMixin,
    ProducersMixin,
    SelectorStepsMixin,
    CalibratorsMixin,
    ShiftTask,
):
    """
    Base task to load histogram and reduce them to used information.
    An exemplary implementation of how to handle the inputs in a run method can be
    found in columnflow/tasks/histograms.py
    """
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # upstream requirements
    reqs = Requirements(
        MergeHistograms=MergeHistograms,
    )

    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("version", "datasets", f"datasets_{self.datasets_repr}")
        return parts

    def requires(self):
        return {
            d: self.reqs.MergeHistograms.req(
                self,
                dataset=d,
                branch=-1,
                _exclude={"branches"},
                _prefer_cli={"variables"},
            )
            for d in self.datasets
        }

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["merged_hists"] = self.requires_from_branch()

        return reqs

    def load_histogram(self, dataset, variable):
        histogram = self.input()[dataset]["collection"][0]["hists"].targets[variable].load(formatter="pickle")
        return histogram

    def reduce_histogram(self, histogram, processes, shifts):
        import hist

        def flatten_nested_list(nested_list):
            return [item for sublist in nested_list for item in sublist]

        # transform into lists if necessary
        processes = law.util.make_list(processes)
        shifts = law.util.make_list(shifts)

        # get all sub processes
        process_insts = list(map(self.config_inst.get_process, processes))
        sub_process_insts = set(flatten_nested_list([
            [sub for sub, _, _ in proc.walk_processes(include_self=True)]
            for proc in process_insts
        ]))

        # get all shift instances
        shift_insts = [self.config_inst.get_shift(shift) for shift in shifts]

        # work on a copy
        h = histogram.copy()

        # axis selections
        h = h[{
            "process": [
                hist.loc(p.id)
                for p in sub_process_insts
                if p.id in h.axes["process"]
            ],
            "shift": [
                hist.loc(s.id)
                for s in shift_insts
                if s.id in h.axes["shift"]
            ],
        }]

        # axis reductions
        h = h[{"process": sum, "shift": sum}]

        return h


class AlphaExtrapolation(HistogramsBaseTask):
    """
    Task to perform alpha extrapolation.
    Read in and plot asymmetry histograms.
    Cut of non-gaussian tails and extrapolate sigma_A( alpha->0 ).
    """
    def output(self) -> dict[law.FileSystemTarget]:
        output = {
            "dummy": self.target("dummy.pickle"),
            # NOTE: do we want to keep plots in a nested directory or in a sibling directory?
            "asym": self.target("plots/asym", dir=True, optional=True),
            "alpha": self.target("plots/alpha", dir=True, optional=True),
            "jer": self.target("plots/jer", dir=True, optional=True),
        }
        return output

    def run(self):
        h_data_all = []
        h_mc_all = []
        for dataset in self.datasets:
            isMC = "qcd" in dataset
            for variable in self.variables:
                h_in = self.load_histogram(dataset, variable)

                # Store all hists in a list to sum over after reading
                if isMC:
                    h_mc_all.append(h_in)
                else:
                    h_data_all.append(h_in)

        h_data = sum(h_data_all)
        h_mc = sum(h_mc_all)  # Not used yet, but keep as reminder

        alpha_bins = [1, 2, 3, 4, 5, 6]
        pt_centers = h_data.axes["dijets_pt_avg"].centers
        asym_centers = h_data.axes["dijets_asymmetry"].centers

        n_eta = len(h_data.axes["probejet_abseta"].centers)
        n_pt = len(h_data.axes["dijets_pt_avg"].centers)

        print("Number \u03B7 bins:", len(h_data.axes["probejet_abseta"].centers))
        print("Number pT bins:", len(h_data.axes["dijets_pt_avg"].centers))
        print("Number A bins:", len(h_data.axes["dijets_asymmetry"].centers))

        proc = 0
        shift = 0
        output = self.output()
        for e in range(n_eta):
            jer = np.array([])
            for p in range(n_pt):
                print(
                    f"\u03B7 bin {e} in ({eta[e]}, {eta[e+1]})",
                    f"pT bin {p} in ({pt[p]}, {pt[p+1]})",
                )

                values = [h_data[hist.loc(a), proc, shift, e, p, :].values() for a in alpha_bins]
                values_inclusive = np.apply_along_axis(np.cumsum, 0, values)

                integral = values_inclusive.sum(axis=1, keepdims=True)
                normalized = values_inclusive / integral
                asym_centers = asym_centers.reshape(1, 160)
                means = np.nansum(asym_centers * normalized, axis=1, keepdims=True)  # Ratio not needed since it is 0/1
                stds = np.sqrt(np.average(((asym_centers - means)**2), weights=normalized, axis=1))
                stds_err = stds / np.sqrt(integral.flatten())  # Not used yet, but keep as reminder

                amax = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
                slope, intercept = np.polyfit(amax[~np.isnan(stds)], stds[~np.isnan(stds)], 1, w=None)
                jer = np.append(jer, intercept * np.sqrt(2) if len(stds[~np.isnan(stds)]) > 1 else None)

            fig, ax = plt.subplots()

            ax.set_xlim(0, 1000)
            ax.plot(pt_centers, jer, marker="o")
            ax.set(**{
                "ylabel": "JER",
                "xlabel": "pt",
            })
            mplhep.cms.label(ax=ax, llabel="CMS Work in progress", data=False)

            plt.tight_layout()
            output = self.output()
            output["jer"].child(f"jer_e{e}.pdf", type="f").dump(plt, formatter="mpl")

        # store some result at the end
        some_result = h_data
        self.output()["dummy"].dump(some_result, formatter="pickle")
